import cv2
with open('data.csv', 'w', newline='') as csvfile:
    filepath = './test2.bmp'
    img = cv2.imread(filepath)
    print(img.shape)
    csvfile.write('0,')
    k = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            highlight = 255 - img[i][j][0]
            csvfile.write(str(highlight))
            k = k + 1
            if k != 784:
                csvfile.write(',')