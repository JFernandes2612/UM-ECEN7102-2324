from LPDetectComponents.LPPredict import predict as predict_features
from LPDetectCharacters.LPPredict import predict as predict_values
from LPDetectComponents.LPExtractFeatures import conv_image
from LPDetectCharacters.LPExtractFeatures import conv_char_img
from PIL import Image

initimgs = conv_image('testimgs/1.jpg')

lps = predict_features(initimgs)

for lp in lps:
    initchars = conv_char_img(lp)

    lpvalue = predict_values(initchars)
    print(f"LP value: {lpvalue}")
    lp.save(f"testimgs/{lpvalue}.jpg")
