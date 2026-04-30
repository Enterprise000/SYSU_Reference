from PIL import Image
image = Image.open('output_stride1_kernel2.ppm')
image.save('result3.jpg', 'JPEG')
