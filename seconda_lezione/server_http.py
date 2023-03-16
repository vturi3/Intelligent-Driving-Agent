import cv2
from flask import Flask, Response, request, jsonify

import numpy as np
import threading
import base64

app = Flask(__name__)

class ServerData:
    def __init__(self) -> None:
        self.__rgb_image = np.zeros((360,640,3), np.uint8)
        self.__depth_image = np.zeros((360,640,3), np.uint8)
        self.__bev_image = np.zeros((360,640,3), np.uint8)

        self.__controls = {"throttle" : 0, "steer" : 0,"brake" : 0}

        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.bev_lock = threading.Lock()
        self.controls_lock = threading.Lock()
        
        self.__record_file = "Speed.txt"
        
    def __record_data(self, data):
        if self.__record_file is not None:
            with open(self.__record_file, "a") as fp:
                fp.write(str(data)+"\n")
                fp.close()
                
    def setRGBImage(self,rgb_image):
        self.rgb_lock.acquire()
        self.__rgb_image = rgb_image
        self.rgb_lock.release()
    
    def getRGBImage(self):
        return self.__rgb_image

    def setDepthImage(self,depth_image):
        self.depth_lock.acquire()
        self.__depth_image = depth_image
        self.depth_lock.release()
    
    def getDepthImage(self):
        return self.__depth_image

    def setBEVImage(self,bev_image):
        self.bev_lock.acquire()
        self.__bev_image = bev_image
        self.bev_lock.release()
    
    def getBEVImage(self):
        return self.__bev_image

    def setControls(self,controls):
        self.controls_lock.acquire()
        self.__controls = controls            
        self.controls_lock.release()
    
    def getControls(self):
        return self.__controls

def threaded(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args)
        thread.setDaemon(True)
        thread.start()
        return thread
    return wrapper
                    
appData = ServerData()

def sendImagesToWeb(getFrame, lock):
    while True:
        try:
            lock.acquire()
            frame = getFrame()
            jpg = cv2.imencode('.jpg', frame)[1]
            lock.release()
            yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+bytearray(jpg)+b'\r\n'
        except Exception as e:
            print("Something went wrong: ", e)
            if lock.locked():
                lock.release()
            yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+bytearray()+b'\r\n'

@app.route('/rgbimage')
def rgbimage():
    global appData
    return Response(sendImagesToWeb(appData.getRGBImage, appData.rgb_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depthimage')
def depthimage():
    global appData
    return Response(sendImagesToWeb(appData.getDepthImage, appData.depth_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bevimage')
def bevimage():
    global appData
    return Response(sendImagesToWeb(appData.getBEVImage, appData.bev_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/controls', methods=['POST'])
def controls():
    global appData
    appData.controls_lock.acquire()
    controls = appData.getControls()
    appData.controls_lock.release()
    return jsonify(controls)

@app.route('/new_frame', methods=['POST']) 
def new_frame():
    global appData
    data = request.json

    if data["type"] == "RGB":
        dataType = np.dtype(data["data"]["dtype"])
        dataArray = np.frombuffer(base64.b64decode(data["data"]["encode"].encode('utf-8')), dataType)
        dataArray = dataArray.reshape(data["data"]["shape"])  
            
        appData.setRGBImage(dataArray)

    elif data["type"] == "Depth":
        dataType = np.dtype(data["data"]["dtype"])
        dataArray = np.frombuffer(base64.b64decode(data["data"]["encode"].encode('utf-8')), dataType)
        dataArray = dataArray.reshape(data["data"]["shape"])  
            
        appData.setDepthImage(dataArray)
    
    elif data["type"] == "Controls":
        appData.setControls(data["data"])
    
    elif data["type"] == "BEV":
        dataType = np.dtype(data["data"]["dtype"])
        dataArray = np.frombuffer(base64.b64decode(data["data"]["encode"].encode('utf-8')), dataType)
        dataArray = dataArray.reshape(data["data"]["shape"])  
            
        appData.setBEVImage(dataArray)

    return jsonify(success=True)

def ajaxClient():
    javascript_code = "<script src=\"https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js\"></script>"\
        "<script type=\"text/javascript\">"\
        "function getControls(){"\
            "$.ajax({"\
                "type: 'POST',"\
                "url: '/controls',"\
                "contentType: 'application/json',"\
                "dataType: 'json',"\
                "success: function(result) {"\
                    "document.getElementById('controls').innerHTML = \" Throttle: \" + result.throttle.toFixed(2) + " \
                                                                    "\" Steer: \" + result.steer.toFixed(2) + " \
                                                                    "\" Brake: \" + result.brake.toFixed(2);" \
                "},"\
                "complete: function(){"\
                    "setTimeout(getControls, 1000);"\
                "}"\
            "});"\
        "};"\
        "getControls();"\
    "</script>"

    return javascript_code

@app.route('/')
def index():
    return "<html>" \
           "<head>"+ajaxClient()+"</head>" \
                "<body>" \
                    "<h1>Carla Simulator</h1>" \
                    "<img src='/rgbimage' style='width: 90%; height: 80%;'/>" \
                    "<h2 id='controls'>CONTROLS : ? ? ?</h1><br>" \
                    "<footer>SimpleVisualizer</footer>" \
                "</body>" \
           "</html>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)