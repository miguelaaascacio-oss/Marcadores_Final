#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp> //necesario para dnn filtro de los candidatos similares
#include "procesa/procesa.h"
#include "marker/reconoce.h" 
#include <sys/socket.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <limits>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <array>      // std::array
#include <cstring>    // memcpy
#include <cmath>      // sqrt, atan2
#include <string>
 	

std::atomic<bool> running(true);
std::atomic<bool> fueraZona(false);
std::atomic<bool> EnRuta(false);

//* --------- ESTRUCTURA DE DATOS --------- /
struct DatosMarcador {
  int id;
  bool detectado;
  int etiqueta = 0;    
  std::vector<cv::Point2f> puntos;
  cv::Point2f centro;
  cv::Matx33d R;
  cv::Vec3d t;
  cv::Vec3d C;  
  double norma;
};

#pragma pack(push,1)
struct MarcadorUDP {
    double ux = 0;
    double uy = 0;
    double uz = 0;
    double norma = -1;
    double yaw = 0;
    double altura = 0;
    double theta = 0;
};
struct PaqueteUDP {
    MarcadorUDP m1;
    MarcadorUDP m2;
    MarcadorUDP m3;
    bool fueraZona = false;
    bool EnRuta    = false;
};
#pragma pack(pop)

// Inversion de eje y para el marco y el triangulo
constexpr float IMG_H = 480.0f;
inline cv::Point2f invertirY(const cv::Point2f& p) {
    return cv::Point2f(p.x, (IMG_H - 1.0f) - p.y);
}

Ort::Session& yolo_session() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    static Ort::Session s(env, "../best.onnx", Ort::SessionOptions{});
    return s;
}

// --- YOLO ---
  
  //sesión de ONNX Runtime con el modelo ya cargado, imagen de la cámara 640x480, en formato BGR opencv

  std::vector<YoloBBox> inferirYOLO(cv::Mat& frame, Ort::Session& session) {
    cv::Mat IMG_yolo;
    cv::resize(frame, IMG_yolo, cv::Size(640, 640)); //rediimensionamos a 640x640 (estandar yolo)
    cv::Mat rgb;
    cv::cvtColor(IMG_yolo, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f); //normalizamos 

// ------------- conversion de HWC a CHW ----------------------
    
    //OpenCV guarda la imagen entrelazada (HWC): los tres canales de cada píxel van juntos.
    //HWC (OpenCV):  R0 G0 B0 | R1 G1 B1 | R2 G2 B2 |
    
    //El modelo espera planar (CHW): primero todos los R, luego todos los G, luego todos los B.
    //CHW (ONNX):    R0 R1 R2 ... | G0 G1 G2 ... | B0 B1 B2

    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch); //separamos la imagen en 3 canales RGB 
    std::vector<float> input_data(3 * 640 * 640); 
    
    // Aplanamiento de canales
    // R desde el elemento 0, G desde el elemento 409,600, B desde el elemento 819,200 total 1,228,800 floats
    for (int c = 0; c < 3; c++)
        memcpy(input_data.data() + c * 640 * 640, ch[c].data, 640 * 640 * sizeof(float));

    std::array<int64_t, 4> shape = {1, 3, 640, 640}; //tensor de entrada 1 imagen 3 canales 640x640
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); //aqui viven los datos 
    
    //se crea tensor de entrada contiene:
    //mem, el puntero a los datos, el número total de elementos, el puntero a la forma y cuántas dimensiones tiene
    auto tensor = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), shape.data(), shape.size());

    const char* input_names[]  = {"images"}; //Los nombres de las capas de entrada y salida de ONNX
    const char* output_names[] = {"output0"};//Los nombres de las capas de entrada y salida de ONNX
    
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1); //se realiza inferencia

    float* data = outputs[0].GetTensorMutableData<float>(); //Puntero al inicio del buffer de la salida
    int num_det = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]; //informacion vector [imagen, detecciones, salidas]
    // data: [ cx0 cy0 w0 h0 conf0 clase0 | cx1 cy1 w1 h1 conf1 clase1 | solo hay una clase, un marcador

    std::vector<YoloBBox> candidatos;
    for (int i = 0; i < num_det; i++) {
        float conf = data[i * 6 + 4];
        if (conf < 0.9f) continue;
        float cx = data[i * 6 + 0], cy = data[i * 6 + 1], w = data[i * 6 + 2], h = data[i * 6 + 3];
        
        // como la salida en un vetor plano encontramos las componentes de esta manera
	//índice:  0   1   2   3   4     5      6   7   8   9   10    11    ...
	//data: [ a   b   c   d  conf  clase | a   b   c   d  conf  clase | ... ]
	//	└──────── fila 0 ────────┘ └──────── fila 1 ────────┘
        
        
        //calculamos el bounding box recortando si se sale de los limites y reescalando el boundingbox de 
        // una imagen de 640 x 640 a 640 x 480

        const float sy = 480.0 / 640.0f;   // 640x480 -> 0.75
        
        float x1 = std::max(0.0f,   cx - w/2 - 5);
        float y1 = std::max(0.0f,   cy - h/2 - 5) ;
        float x2 = std::min(640.0f, cx + w/2 + 5);
        float y2 = std::min(640.0f, cy + h/2 + 5);

        YoloBBox box;
        box.x = x1;                // esquina superior izquierda
        box.y = y1 * sy;
        box.w = (x2 - x1);         // ancho
        box.h = (y2 - y1) * sy;         // alto
        box.confidence = conf;
        box.valid = true;
        candidatos.push_back(box);
    }
    
    
    //return candidatos; //candidatos = [ box1, box2, ... ] que pasaron el umbral no tiene NMS                           (Non-Maximum Suppression y manda muchas cajas identicas usamos el de opencn dnn <opencv2/dnn.hpp>
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    
    for (const auto& c : candidatos) {
        rects.emplace_back(cvRound(c.x), cvRound(c.y), cvRound(c.w), cvRound(c.h));
        scores.push_back(c.confidence);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(rects, scores, 0.9f , 0.5f, idx); // idx = posicion de candidato que sobreviven al NMS

    std::vector<YoloBBox> keep;
    for (int i : idx) keep.push_back(candidatos[i]);
        return keep;                 // aqui estan los candidatos no repetidos
   }


//* --------- PROYECCIÓN (2D HOMOGRAFÍA) --------- /

void proyectarMarcador(cv::Mat& img, const DatosMarcador& datos, std::atomic<bool>& zonaGlobal, std::atomic<bool>& enRuta) 
{
    if (!datos.detectado || datos.puntos.size() != 6) {
    return; 
    }

    constexpr float xmin = 30.0f;
    constexpr float xmax = 610.0f;

    constexpr float ymin = 30.0f;
    constexpr float ymax = 450.0f;
    
    constexpr float xmin2 = 190.0f;
    constexpr float xmax2 = 450.0f;

    constexpr float ymin2 = 110.0f;
    constexpr float ymax2 = 330.0f;
    
    //x ∈ [30 , 610]
    //y ∈ [30 , 450]

    bool esteFuera = false;
    bool marcadorEnAreaFoco = false;
        
    for (const auto& p : datos.puntos)
    {
       if (p.x < xmin || p.x > xmax ||
           p.y < ymin || p.y > ymax)
       {
           esteFuera = true;
       }
       if (p.x >= xmin2 && p.x <= xmax2 &&
           p.y >= ymin2 && p.y <= ymax2)
       {
           marcadorEnAreaFoco = true;
       }
 
    }
    
    if (marcadorEnAreaFoco)              
    	enRuta.store(true);
    if (marcadorEnAreaFoco && esteFuera) 
    	zonaGlobal.store(true);
    
    // Dibujar rectangulo de zona segura
	cv::rectangle(img,
		      cv::Point(30,30),
		      cv::Point(610,450),
		      zonaGlobal.load() ? cv::Scalar(0,0,255)
		                       : cv::Scalar(0,255,0),
		      2);
		      
    // Dibujar rectangulo de ruta
	cv::rectangle(img,
		      cv::Point(190,110),
		      cv::Point(450,330),
		      enRuta.load() ? cv::Scalar(0,255,0)
		                       : cv::Scalar(0,0,255),
		      2);

    std::vector<cv::Point2f> marcador = {
        {255, 0},       // P1 - Esquina superior derecha (Y invertido: 255-255=0)
        {0, 0},         // P2 - Esquina superior izquierda (Y invertido: 255-255=0)
        {90, 56},       // P3 - Punto medio izquierdo (Y invertido: 255-199=56)
        {165, 56},      // P4 - Punto medio derecho (Y invertido: 255-199=56)
        {0, 255},       // P5 - Esquina inferior izquierda (Y invertido: 255-0=255)
        {255, 255}      // P6 - Esquina inferior derecha (Y invertido: 255-0=255)
    };

    //std::vector<cv::Point2f> imagen = datos.puntos;
    std::vector<cv::Point2f> imagen = datos.puntos;
    cv::Mat H = cv::findHomography(marcador, imagen, 0);
    if (H.empty()) {
        return;  // Salir si no se pudo calcular la homografía
    }
    
    std::vector<cv::Point2f> contorno = {
        {255, 0},   // Esquina superior derecha
        {0, 0},     // Esquina superior izquierda
        {0, 255},   // Esquina inferior izquierda
        {255, 255}  // Esquina inferior derecha
    };

    std::vector<cv::Point2f> contorno_img;
    cv::perspectiveTransform(contorno, contorno_img, H);
    
    cv::Scalar colorContorno = enRuta.load() ? cv::Scalar(0, 255, 255) : 
                               (datos.id == 2 ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 0));

    for (int i = 0; i < 4; i++) {
    cv::Point2f p0 = invertirY(contorno_img[i]);
    cv::Point2f p1 = invertirY(contorno_img[(i+1)%4]);

    cv::line(img,
             p0,
             p1,
             colorContorno,
             2);
	}

    std::vector<cv::Point2f> tri = {
        {90, 56},           // Punto medio izquierdo
        {165, 56},          // Punto medio derecho
        {127.5f, 127.5f}    // Punto central superior (mitad de 255 = 127.5)
    };

    std::vector<cv::Point2f> tri_img;
    cv::perspectiveTransform(tri, tri_img, H);

	for (int i = 0; i < 3; i++) {
	    cv::Point2f p0 = invertirY(tri_img[i]);
	    cv::Point2f p1 = invertirY(tri_img[(i+1)%3]);

	    cv::line(img,
		     p0,
		     p1,
		     cv::Scalar(0,0,255),
		     2);
	}
	
	cv::Point2f centroide = invertirY(tri_img[2]); // Usamos la punta del triángulo como ancla
        cv::putText(img, "Marcador: " + std::to_string(datos.etiqueta), cv::Point(centroide.x - 20, centroide.y - 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, colorContorno, 2);
}

//* ------------------ FUNCIÓN PRINCIPAL ------------------ /
int main(int /*argc*/, char** /*argv*/)
{

    // --- Socket UDP hacia Python ---
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in servaddr{};
    servaddr.sin_family      = AF_INET;
    servaddr.sin_port        = htons(5005);
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");


    int frame_id = 0;
    Ort::Session& session = yolo_session(); //inicia YOLO
    
    
    std::string pipeline =
        "udpsrc port=11111 buffer-size=2048000 ! "      // 1. Recibe paquetes UDP del Tello
        "h264parse ! "                                   // 2. Reensambla los trozos de video H.264
        "avdec_h264 max-threads=1 ! "                    // 3. Decodifica H.264 a imagen (1 hilo = menos retraso)
        "videoconvert ! "                                // 4. Cambia el formato de color YUV → BGR
        "video/x-raw,format=BGR ! "                      // 5. Obliga a que el formato final sea BGR
        "appsink drop=true max-buffers=1 sync=false";     // 6. Entrega solo el frame más nuevo, sin esperar

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);   // Abre el pipeline con GStreamer
    
    
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el stream con GStreamer." << std::endl;
        return -1;
    }
    
    cv::Mat frame;

    while (running.load()) {
        
        cap >> frame; 
        if (frame.empty()) break;
        if (cv::waitKey(1) == 27) break; // ESC para salir

        // ---------- Preprocesado ----------
        cv::Mat grayFull;
        cv::resize(frame, frame, cv::Size(640, 480));
        cv::cvtColor(frame, grayFull, cv::COLOR_BGR2GRAY);

        // ---------- Detección  ----------
        // Aqui se llama a la red YOLO para detectar los marcadores
        std::vector<YoloBBox> cajas = inferirYOLO(frame, session);  //se manda a llamar YOLO
         //cout << "YOLO cajas: " << candidatos.size() << endl; //ver las cajas encontradas arriba del umbral
      
        // Ver las caja de Yolo comentar todo el ciclo para ya no verlas 
        for (const auto& c : cajas) {
             cv::Rect r(cvRound(c.x), cvRound(c.y), cvRound(c.w), cvRound(c.h));
             cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 2);            

            char txt[32];
            snprintf(txt, sizeof(txt), "%.2f", c.confidence);
            cv::putText(frame, txt,
                cv::Point(r.x, std::max(15, r.y - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }

        // para cada marcador detectado se llama a homografias en su caja contenedora
        std::vector<MarcadorHomografia> detectados;
        
        for (const auto& c : cajas) {
            std::array<int, 4> caja = { static_cast<int>(c.x), static_cast<int>(c.y),
                                        static_cast<int>(c.w), static_cast<int>(c.h) };
                                
            // c.x c.y c.w c.h en opencv x,y es la esquina superior izquierda y w en ancho y h el alto de la caja
            
            std::vector<MarcadorHomografia> det =
                obtener_homografias(frame, grayFull, caja, 115);
            detectados.insert(detectados.end(), det.begin(), det.end());
        }
        
        // ---------- Construcción de DatosMarcador ----------
        std::vector<DatosMarcador> marcadores_procesados;
        marcadores_procesados.reserve(detectados.size());

        for (const auto& m : detectados) {
            DatosMarcador d;
            d.id        = m.id;
            d.detectado = true;
            d.puntos.assign(m.puntos.begin(), m.puntos.end());
            
            d.R = m.R;
            d.t = m.t;
                        
            d.centro = cv::Point2f(0.f, 0.f); ;
            for (const auto& p : d.puntos) d.centro += p;
            if (!d.puntos.empty()) d.centro *= 1.0f / static_cast<float>(d.puntos.size());

            //Marcador centrado en la camara vector apuntador U = -C
            d.C = -d.R.t() * d.t;
            
            double cx = d.C[0];
            double cy = d.C[1];
            double cz = d.C[2];
            
            d.norma = std::sqrt((cx * cx) + (cy * cy) + (cz * cz));

            marcadores_procesados.push_back(d);
        }
            
        // ---------- Ordenación Y etiquetas ----------
        std::sort(marcadores_procesados.begin(), marcadores_procesados.end(),
            [&](const DatosMarcador& a, const DatosMarcador& b) {
            return a.centro.y > b.centro.y;
        });
        
        // ---------- Preparar paquete UDP ----------
        PaqueteUDP paquete;         // default: todo en 0, norma = -1
        paquete.m1.norma = -1;
        paquete.m2.norma = -1;
        paquete.m3.norma = -1;

        fueraZona.store(false);
        EnRuta.store(false);

        // ---------- Iterar marcadores ----------
        int etiqueta_actual = 1;
    
        for (auto& mp : marcadores_procesados) {
            if (etiqueta_actual > 3) break; 
            mp.etiqueta = etiqueta_actual;

            proyectarMarcador(frame, mp, fueraZona, EnRuta);

            // >>> POSE <<< aqui debe calcular se el U = -Cdron que es el que se va a enviar por cada marcador
           
            // >>> CÁLCULO DE POSE <<<
            // t en cm -> m
            double tx = mp.t[0] / 100.0;
            double ty = mp.t[1] / 100.0;
            double tz = mp.t[2] / 100.0;
 
            double norma = std::sqrt(tx*tx + ty*ty + tz*tz);
            double ux = (norma > 1e-9) ? tx / norma : 0.0;
            double uy = (norma > 1e-9) ? ty / norma : 0.0;
            double uz = (norma > 1e-9) ? tz / norma : 0.0;

            double nx = mp.R(0, 2);
            double ny = mp.R(1, 2);
            double nz = mp.R(2, 2);
            double altura = std::abs(nx*tx + ny*ty + nz*tz); 

	    
            // ángulo de la cámara alrededor del marcador, en el plano del marcador
            double px = -(mp.R(0,0)*tx + mp.R(1,0)*ty + mp.R(2,0)*tz);   // 
            double py = -(mp.R(0,1)*tx + mp.R(1,1)*ty + mp.R(2,1)*tz);
            double theta = std::atan2(py, px);
 
            // yaw alrededor del eje Z a partir de R
            double yaw = std::atan2(tx, -tz);   // 
                                    
            // >>> FIN CÁLCULO DE POSE <<<
	    
	    MarcadorUDP slot;
            slot.ux = ux; slot.uy = uy; slot.uz = uz;
            slot.norma = norma;
            slot.yaw = yaw;
            slot.altura = altura;
            slot.theta = theta;

            if      (etiqueta_actual == 1) paquete.m1 = slot;
            else if (etiqueta_actual == 2) paquete.m2 = slot;
            else if (etiqueta_actual == 3) paquete.m3 = slot;

            // Texto AR
            int oy = (etiqueta_actual - 1) * 140;
            char buf[128];
            snprintf(buf, sizeof(buf), "E%d Profundidad: %.2f m", mp.etiqueta, -tz);
            cv::putText(frame, buf, {30, 30 + oy},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,255}, 2);
            snprintf(buf, sizeof(buf), "X:%.2f Y:%.2f", tx, ty);
            cv::putText(frame, buf, {30, 60 + oy},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);

            etiqueta_actual++;
        }
        
        paquete.fueraZona = fueraZona.load();
        paquete.EnRuta    = EnRuta.load();

        // ---------- Envío UDP ----------
        sendto(sockfd, &paquete, sizeof(PaqueteUDP), 0,
               (const struct sockaddr*)&servaddr, sizeof(servaddr));
               
        // ---------- Visualización ----------
        cv::imshow("Marcador Proyectado", frame);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') running = false;

        frame_id++;
    }

    running = false;
    if (sockfd >= 0) close(sockfd);
    cap.release();
    cv::destroyAllWindows();
    printf("\nPrograma finalizado.\n");
    return 0;
}
