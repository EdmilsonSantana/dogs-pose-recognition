from bing_image_downloader import downloader
import sys

downloader.download(sys.argv[1], limit=100,  output_dir='dataset2',
                    adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
