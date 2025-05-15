import json 
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from ocr import OCRNeuralNetwork


nn = OCRNeuralNetwork(
    num_hidden_nodes=25,
    data_matrix=[],  
    data_labels=[],
    train_indices=[],
    use_file=True
)
nn._load()  # Load saved weights if available

class RequestHandler(BaseHTTPRequestHandler):
    MIME_TYPES = {
        ".html": "text/html",
        ".js": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".ico": "image/x-icon"
    }

    def do_GET(self):
        path = self.path if self.path != "/" else "/ocr.html"
        filepath = os.path.join(os.path.dirname(__file__), path.lstrip("/"))
        
        if os.path.isfile(filepath):
            ext = os.path.splitext(filepath)[1]
            mime_type = self.MIME_TYPES.get(ext, "application/octet-stream")

            self.send_response(200)
            self.send_header("Content-type", mime_type)
            self.end_headers()

            with open(filepath, "rb") as file:
                self.wfile.write(file.read())
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        response_code = 200
        response = ""
        try:
            var_len = int(self.headers.get('Content-Length'))
            content = self.rfile.read(var_len)
            payload = json.loads(content)
            
            if payload.get('train'):
                for data in payload['trainArray']:
                    nn.train_on_instance(data) 
                nn.save()  
                response = {"message": "Training completed successfully for batch size 1"}
            elif payload.get('predict'):
                try:
                    result = nn.predict(payload['image'])  # expects 'image' key
                    response = {
                        "type": "test",
                        "result": result
                    }
                except Exception as e:
                    print(f"Prediction error: {e}")
                    response_code = 500
            else:
                response_code = 400
        except Exception as e:
            print(f"POST error: {e}")
            response_code = 500

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server = HTTPServer(('localhost', 8080), RequestHandler)
    print(f"Server running on port {port}...")
    server.serve_forever()

if __name__ == "__main__":
    run()