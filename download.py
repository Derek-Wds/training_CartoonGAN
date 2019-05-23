import requests

# grab photos from Pixabay
class Pixabay:
    def __init__(self, key, min_width=300, min_height=300, query=''):
        self.url = 'https://pixabay.com/api'
        self.params = {'key': key, 'image_type': 'photo', 
                        'pretty': 'false', 'min_width': min_width,
                        'min_height': min_height, 'per_page': 200
                        }
        if len(query) > 0:
            self.params['query'] = query
        self.page = 1
    
    def search(self):
        r = requests.get(self.url, params=self.params)
        hits = r.json()['hits']
        return [i['largeImageURL'] for i in hits]

    def set_query(self, query):
        self.page = 1
        if len(query) > 0:
            self.params['query'] = query

counter = 0

key = "7429820-c0f17225d11abf9ffe919ad24"
p = Pixabay(key)

for i in range(10):
    res = p.search()
    for img in res:
        img_req = requests.get(img)
        if img_req.status_code == 200:
            with open("./"+str(counter)+'.jpg', 'wb') as f:
                f.write(img_req.content) 
                counter += 1

# grab cartoon photos
