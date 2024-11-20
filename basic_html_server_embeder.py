import http.server
import socketserver
import json
import sentence_transformers
import torch
from toml_loader import get_config

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("GPU is:", torch.cuda.get_device_name(0))
else:
    #TODO: install cuda
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
    use_cuda = False

model = sentence_transformers.SentenceTransformer("intfloat/multilingux1al-e5-large", device=device)
print("model loaded")
class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        print("processing batch")
        texts = data["texts"]
        print(f"recieved {len(texts)}")
        result = {}
        for text, emb in zip(texts, model.encode([f"{data['pre']}: {x}" for x in texts], normalize_embeddings=True)):
            result[text] = list(map(float, emb))
        print(f"sent {len(result.keys())}")
        if len(result.keys()) < len(texts):
            print("SENT OUT LESS THEN RECIEVED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))

if __name__ == "__main__":
    PORT = get_config()["embeder"]["port"]
    Handler = MyHttpRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()