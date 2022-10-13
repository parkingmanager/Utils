# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:27:35 2022

@author: Parking Manager 
"""


import os
import json
import math
from math import dist
import numpy as np
import re
import cv2 as cv
import time
import shutil
from random import randint

"""
Funciones
"""



def format_image(image):
    
    SIZE = 299

    # Imagen vacia 
    blank_image = 0 * np.ones((SIZE,SIZE,3), np.uint8)
    
    # Proyectar imagen de menores dimensiones
    img = image
    
    # Shape: h, w
    img_h = np.shape(img)[0]
    img_w = np.shape(img)[1]
    img_aspect_ratio = img_h/img_w
           
    # Debe redimensionar < downscale
    if img_h > SIZE or img_w > SIZE:
    
        if img_h >= img_w:
            img_h = SIZE
            img_w = int(img_h / img_aspect_ratio) 
        else: 
            img_w = SIZE
            img_h = int(img_w / img_aspect_ratio) 
    
        img = cv.resize(img, (img_w, img_h), interpolation = cv.INTER_AREA)
    
    # Separa los canales. 
    b_blank, g_blank, r_blank = cv.split(blank_image) 
    b_img, g_img, r_img = cv.split(img) 
    
    # Itera sobre imagen en los 3 canales
    for i in range(len(b_img)):
        for j in range(len(b_img[i])):
            b_blank[i,j] = b_img[i,j]      
    
    for i in range(len(g_img)):
        for j in range(len(g_img[i])):
            g_blank[i,j] = g_img[i,j]   
            
    for i in range(len(r_img)):
        for j in range(len(r_img[i])):
            r_blank[i,j] = r_img[i,j]   
    
    # Junta los canales y las imagenes 
    merged = cv.merge([b_blank, g_blank, r_blank])
    
    return merged

def generate_videos_to_process(videos_path):
    
    output: list = []
        
    # Lista de carpetas de videos
    parkings = os.listdir(videos_path)
    
    for par in parkings:
        # Lista de perspectivas en el parqueadero
        perspectives = os.listdir(videos_path + f"\{par}")
        
        for per in perspectives:
            
            # Extrae archivos y filtra unicamente imagenes jpg
            files = os.listdir(videos_path + f"\{par}\{per}")
            videos = list(filter(lambda a: ".mp4" in a, files))
            
            for vi in videos: 
                                 
                
                video_to_process = {
                        "video" : videos_path + f"\{par}\{per}\{vi}",
                        "config" : (videos_path + f"\{par}\{per}\{vi}").replace(".mp4", ".json"),
                        "folder" : (videos_path + f"\{par}\{per}\{vi}").replace(".mp4", ""),
                        "processed" : videos_path + f"\{par}\{per}\procesado.txt"
                    }
                
                if os.path.exists(video_to_process["processed"]):
                    print("La perspectiva ya fue procesada...")
                    print(video_to_process["processed"])
                
                elif os.path.isfile(video_to_process["video"]) and os.path.isfile(video_to_process["config"]):
                    output.append(video_to_process)
                    
                    # Crea o limpia la carpeta para guardar el dataset
                    if not os.path.exists(video_to_process["folder"]):
                        os.makedirs(video_to_process["folder"])  
                        print(f'El directorio {video_to_process["folder"]} fue creado')
                    else:    
                        shutil.rmtree(video_to_process["folder"])
                        os.makedirs(video_to_process["folder"])  
                        print(f'El directorio {video_to_process["folder"]} fue vaciado')
                        
                else:
                    print(f"El video no puede ser procesado...\n\r {video_to_process}")
    return output

def clean_coordinates(coord):
    return re.sub("[^\w\s]", "", coord)

def load_json(file: str):
    
    print(file)
    
    # JSON file
    f = open (file, "r")
    data = json.loads(f.read())
    f.close()
    
    return data

def get_coordinates(json_data: dict):
    
    coords = []
    
    # h = 400
    # w = 200
    
    spaces = json_data['EspacioDelimitadoes']
       
    for s in spaces:                      
               
        coord1 = np.fromstring(clean_coordinates(s["Coord1"]), dtype=int, sep='  ')
        coord2 = np.fromstring(clean_coordinates(s["Coord2"]), dtype=int, sep='  ')
        coord3 = np.fromstring(clean_coordinates(s["Coord3"]), dtype=int, sep='  ')
        coord4 = np.fromstring(clean_coordinates(s["Coord4"]), dtype=int, sep='  ')

        #coords.append([coord1, coord2, coord3, coord4])        
        coords.append({"coords":[coord1, coord2, coord3, coord4], "type":  s["Tipo"]})
        
    return coords

"""
Variables
"""

# Listado de videos a procesar...
videos_to_process : list = []

# Path base de la carpeta Dataset
base = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Path de la carpeta de videos
videos_path = base + '\Videos'


"""
Lógica
"""
# Extrae videos a procesar
videos_to_process = generate_videos_to_process(videos_path)

for v in videos_to_process:
    
    # Carga información de coordenadas json
    json_data = load_json(v["config"])   
    
    coords = get_coordinates(json_data)
    
    cap = cv.VideoCapture(v["video"])
    
    cont = 0
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        
        if not ret:
            print("Ok, el video finalizó, siguiente ...")
            break
        
        # General
        height, width = frame.shape[:2]
        
        # Transf perspectiva
        h = 400
        w = 200       
                
        if cont % 50 == 0:
            
            warp_number = 0
            
            for coord in coords:
                          
                c = coord["coords"]
                
                pts1 = np.float32([[c[0][0], c[0][1]],
                                   [c[1][0], c[1][1]],
                                   [c[3][0], c[3][1]],
                                   [c[2][0], c[2][1]]])
                
                p1 = (c[0][0], c[0][1])
                p2 = (c[1][0], c[1][1])
                p3 = (c[3][0], c[3][1])
                p4 = (c[2][0], c[2][1])
                
                # Horizontales
                dp1p2 = dist(p1,p2)
                dp3p4 = dist(p3,p4)
                dw = int((dp1p2 + dp3p4)/2)
                
                # Verticales
                dp1p3 = dist(p1,p3)
                dp2p4 = dist(p2,p4)       
                dh = int((dp1p3+dp2p4)/2)                
                
                # Imagen normal
                pts2 = np.float32([[0, 0],[dw, 0],[0, dh],[dw, dh]])
                m = cv.getPerspectiveTransform(pts1, pts2)
                warped = cv.warpPerspective(frame, m, (dw, dh))
                
                # Imagen 200x400
                pts_200x400 = np.float32([[0, 0],[w, 0],[0, h],[w, h]])
                m_200x400 = cv.getPerspectiveTransform(pts1, pts_200x400)
                warped_200x400 = cv.warpPerspective(frame, m_200x400, (w, h))

                # Imagenes bw
                warped_bw = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
                warped_bw_200x400 = cv.cvtColor(warped_200x400, cv.COLOR_BGR2GRAY)
                
                
                cv.imwrite(f'{v["folder"]}\\{cont}_{warp_number}_{coord["type"]}.png', warped)
                cv.imwrite(f'{v["folder"]}\\{cont}_{warp_number}_{coord["type"]}_220x400.png', warped_200x400)
                
               	if randint(0, 10) > 5:                       
                       cv.imwrite(f'{v["folder"]}\\{cont}_{warp_number}_bw_{coord["type"]}.png', warped_bw)
                if randint(0, 10) > 5:                       
                       cv.imwrite(f'{v["folder"]}\\{cont}_{warp_number}_bw_{coord["type"]}_200x400.png', warped_bw_200x400)
                             
                warp_number += 1                        
        
        cont += 1                
      
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
