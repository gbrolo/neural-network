from google_images_download import google_images_download

resp = google_images_download.googleimagesdownload()

# get images
def get_images(keywords):
    arguments = {"keywords":keywords,"limit":50,"print_urls":True}
    paths = resp.download(arguments)
    print(paths)

# get_images('black and white circle jpg')
# get_images('black and white square jpg')
# get_images('black and white triangle jpg')
# get_images('black and white egg jpg')
# get_images('black and white tree jpg')
# get_images('black and white house jpg')
# get_images('black and white smiley face jpg')
# get_images('black and white sad face jpg')
# get_images('black and white question mark jpg')
# get_images('black and white mickey mouse face shadow jpg')
# get_images('black and white tree sketch jpg')
get_images('circle sketch black and white jpg')

