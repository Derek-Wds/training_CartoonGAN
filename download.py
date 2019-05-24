import requests
from urllib import request
from pprint import pprint
from google_images_download import google_images_download 

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


# function to dowload the cartoon images from google
response = google_images_download.googleimagesdownload() 

def downloadimages(query): 
    # keywords is the search query 
    # format is the image file format 
    # limit is the number of images to be downloaded 
    # print urs is to print the image file url 
    # size is the image size which can 
    # be specified manually ("large, medium, icon") 
    # aspect ratio denotes the height width ratio 
    # of images to download. ("tall, square, wide, panoramic") 
    arguments = {"keywords":query,
                    "format": "jpg",
                    "limit":100,
                    "print_urls":True,
                    "size": "medium",
                    "aspect_ratio": "panoramic",
                    "chromedriver":"/home/dingsu/chromedriver", # One needs to change this in order to make the selenium work
                    "image_directory":"dataset/cartoon_imgs"}
    try: 
        response.download(arguments) 
    
    # Handling File NotFound Error	 
    except FileNotFoundError: 
        arguments = {"keywords": query, 
                    "format": "jpg", 
                    "limit":100, 
                    "print_urls":True, 
                    "size": "medium",
                    "chromedriver":'/home/dingsu/chromedriver', # One needs to change this in order to make the selenium work
                    "image_directory":"dataset/cartoon_imgs"} 
                    
        # Providing arguments for the searched query 
        try: 
            # Downloading the photos based 
            # on the given arguments 
            response.download(arguments) 
        except: 
            pass


if __name__ == "__main__":
    # creating object 
    print('==========================================================================')
    print('Downloading the studio ghibli cartoon images from google')
    print('==========================================================================')

    # Driver Code 
    downloadimages("miyazaki spirited away wallpaper")
    downloadimages("miyazaki my neighbor totoro wallpaper")
    downloadimages("miyazaki howl's moving castle wallpaper")
    downloadimages("miyazaki castle in the sky wallpaper")
    downloadimages("miyazaki ponyo on the cliff wallpaper")
    downloadimages("studio ghibli cartoon images")


    print('==========================================================================')
    print('Cartoon images download completed!')
    print('==========================================================================')


    print()
    print()


    print('==========================================================================')
    print('Download photos from pixabay')
    print('==========================================================================')
    
    counter = 0
    key = "7429820-c0f17225d11abf9ffe919ad24"
    p = Pixabay(key)

    for i in range(100):
        res = p.search()
        print('Downloading round {} images!'.format(i+1))
        for img in res:
            img_req = requests.get(img)
            if img_req.status_code == 200:
                with open("./dataset/photo_imgs/"+str(counter)+'.jpg', 'wb') as f:
                    f.write(img_req.content) 
                    counter += 1
                    print(str(counter) + ".jpg" + ' has been saved!')

    print('==========================================================================')
    print('Photos download completed!')
    print('==========================================================================')

    print()
    print()
    print("ATTENTION! Please remember to move the photos to 'dataset' folder and creat 3 sub folders in it:" + \
    " 'cartoon_imgs' and 'smooth_cartoon_imgs'")