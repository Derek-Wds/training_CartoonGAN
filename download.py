import requests, argparse
import os
import shutil
from PIL import Image

# grab photos from Pixabay
class Pixabay:
    def __init__(self, key, min_width=300, min_height=300, query='', page=1):
        self.url = 'https://pixabay.com/api'
        self.params = {'key': key, 'image_type': 'photo', 
                        'pretty': 'false', 'min_width': min_width,
                        'min_height': min_height, 'per_page': 200, 'page': page
                        }
        if len(query) > 0:
            self.params['q'] = query
        self.page = page
    
    def search(self):
        r = requests.get(self.url, params=self.params)
        hits = r.json()['hits']
        return [i['largeImageURL'] for i in hits]

    def set_query(self, query):
        self.page = page
        if len(query) > 0:
            self.params['query'] = query


def main():

    DEFAULT_SIZE = (256, 256)

    print('==========================================================================')
    print('Check if source video exists.')
    print('==========================================================================')
    if(not os.path.isdir('video')):
        print('Be sure to include video files inside video folder')
        raise Exception("No Video")
    else:
        print('video folder exists!')
    print()

    print('==========================================================================')
    print('Create dataset folder.')
    print('==========================================================================')

    if (os.path.isdir('dataset')):
        print("Clear previous dataset")
        shutil.rmtree('dataset')
    
    print('Creating folders ... ...')
    os.mkdir('dataset')
    os.mkdir('dataset/photo_imgs')
    os.mkdir('dataset/cartoon_imgs')
    os.mkdir('dataset/smooth_cartoon_imgs')
    print('Folders created!')
    print()

    print('==========================================================================')
    print('Cartoon images process started!')
    print('==========================================================================')

    # files = ['video/'+f for f in os.listdir('video') if os.path.isfile('video/'+f)]
    # file_counter = 0
    # for video_file in files:
    #     command = 'ffmpeg -i "{}" -start_number 0 -ss 00:10:00 -vf fps=1 -s 256x256 "dataset/cartoon_imgs/${}-%05d.jpg"'.format(video_file, file_counter)
    #     os.system(command)
    # print()

    print('==========================================================================')
    print('Cartoon images process completed!')
    print('==========================================================================')


    print()
    print()


    print('==========================================================================')
    print('Download photos from pixabay')
    print('==========================================================================')
    
    counter = 0
    key = "7429820-c0f17225d11abf9ffe919ad24"
    query_list=['human+face','mountains','animals','sports','fashion','food','cars','technology','places','music','industry']

    for i in range(10):
        counter += 1
        for j in range(1, 4):
            if counter % 251 == 0:
                break
            p = Pixabay(key, query=query_list[i], page=j)
            res = p.search()
            print('Downloading round {} images!'.format(i+1))
            for img in res:
                img_req = requests.get(img)
                if img_req.status_code == 200:
                    file_name = "./dataset/photo_imgs/"+str(counter)+'.jpg'
                    try:
                        with open(file_name, 'wb') as f:
                            f.write(img_req.content) 
                            img = Image.open(file_name)
                            img = img.resize(DEFAULT_SIZE)
                            img.save(file_name)
                            print(str(counter) + ".jpg" + ' has been saved!')
                            counter += 1
                    except:
                        print('Error occurred during processing, continue downloading ... ...')

    print('==========================================================================')
    print('Photos download completed!')
    print('==========================================================================')

    print()
    print()
    print("ATTENTION! Please remember to move the photos to 'dataset' folder and creat 3 sub folders in it:" + \
    " 'cartoon_imgs' and 'smooth_cartoon_imgs'")

if __name__ == "__main__":
    main()