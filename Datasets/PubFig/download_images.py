"""
Python script to download images of the PubFig dataset.
"Attribute and Simile Classifiers for Face Verification,"
Neeraj Kumar, Alexander C. Berg, Peter N. Belhumeur, and Shree K. Nayar,
International Conference on Computer Vision (ICCV), 2009.

Author: Cristiano Patrício
E-mail: cristiano.patricio@ubi.pt
University of Beira Interior, Portugal
"""

import urllib
import re
import urllib.request
import socket
import http
import argparse
import time

# Set TimeOut
socket.setdefaulttimeout(10)

# Args
parser = argparse.ArgumentParser(description='Download PubFig Images Script')
parser.add_argument('--url-file-dir', type=str, default='',
                    help='Directory path containing dev_urls.txt / eval_urls.txt file.')
parser.add_argument('--save-dir', type=str, default='',
                    help='Directory path to save images.')


##########################################################################
# AUXILIARY FUNCTIONS
##########################################################################

def get_url(file):
    urls = []
    name_url_dict = dict()
    with open(file, 'r') as f:
        lines = f.readlines()
        for l in lines[2:]:
            line = re.split("\t", l)
            # url is in the 2 position
            url = str(line[2])
            urls.append(url)
            # image name line 0 + 1
            pre_name = line[0].replace(" ", "_")
            id = int(line[1])

            if id < 10:
                name = pre_name + str("_") + str('000') + str(id)
            elif 10 <= id < 100:
                name = pre_name + str("_") + str('00') + str(id)
            elif 100 <= id < 1000:
                name = pre_name + str("_") + str('0') + str(id)
            else:
                name = pre_name + str("_") + str(id)

            name_url_dict[url] = str(name)

    return urls, name_url_dict


def download_image_to_folder(path_folder, urls, name_url):
    success = 0
    errors = 0
    tic = time.time()
    for url in urls:
        try:
            print("[INFO]: Trying to download {0}.jpg ... ".format(name_url[url]))
            urllib.request.urlretrieve(url, path_folder + name_url[url] + ".jpg")
            print("[INFO]: Response: SUCCESS!")
            success += 1
        except urllib.error.URLError:
            print("[INFO]: Response: INVALID URL.")
            errors += 1
            pass
        except urllib.error.HTTPError:
            print("[INFO]: Response: HTTP Error.")
            errors += 1
            pass
        except http.client.HTTPException:
            print("[INFO]: Response: HTTP Exception.")
            errors += 1
            pass
        except socket.error:
            print("[INFO]: Response: Socket Error")
            errors += 1
            pass
    toc = time.time()

    elapsed_time = (toc - tic) / 60 / 60

    return success, errors, elapsed_time


if __name__ == "__main__":
    args = parser.parse_args()
    dir_path = args.save_dir
    file_path = args.url_file_dir
    urls, name_url = get_url(file_path)
    n_success, n_errors, elapsed_time = download_image_to_folder(dir_path, urls, name_url)
    n_images = len(urls)
    print("----------------------- COMPLETED! -----------------------")
    print("[INFO]: Process completed in {:.2f} hours.".format(elapsed_time))
    print("[INFO]: Successfully downloaded {0} of {1} images.".format(n_success, n_images))
    print("[INFO]: Total errors: {0}.".format(n_errors))
