from LPDetectComponents.LPPredict import predict as predict_features
from LPDetectCharacters.LPPredict import predict as predict_values
from LPDetectComponents.LPExtractFeatures import conv_image
from LPDetectCharacters.LPExtractFeatures import conv_char_img
from PIL import Image

initimgs = conv_image('testimgs/1.jpg')
[k.save(f"ci{i}.jpg") for i,k in enumerate(initimgs)]

lps = predict_features(initimgs)

for lp in lps:
    initchars = conv_char_img(lp)
    [k.save(f"ch{i}.jpg") for i,k in enumerate(initchars)]

    lpvalue = predict_values(initchars)
    print(f"LP value: {lpvalue}")
    lp.save(f"testimgs/{lpvalue}.jpg")
