from run import get_result

# 입력 이미지는 정사각형
# 정규 규격은 28*28이나 자동으로 변환하므로 더 커도 상관 없음 (예시 이미지는 128 * 128)
# 굵은 펜으로 해야함

# Shell에서 "python runTest.py [image_path] ./model/trained_model ./model/dump"
# 또는 function 호출로 "main_module('image path', './model/trained_model.h5', './model/dump')"

result = get_result('bee.jpg', 'model1-35/model.h5', 'model1-35/dump')

print(result)
