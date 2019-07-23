# 加载预训练模型和输出模型参数

from vgg.vgg import vgg_16
from vgg import vgg
from tensorflow.contrib import slim
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

# R_MEAN = None
# G_MEAN = None
# B_MEAN = None
# # 减去图片均值
#
# def FCN(inputs, args, is_training, reuse):
#     inputs = inputs - [R_MEAN, G_MEAN, B_MEAN]
#
#     # inputs has shape - Original: [batch, 513, 513, 3]
#
#     VGG = getattr(vgg, args.vgg_model)  # 选择要使用的模型
#     _, end_points = VGG(inputs,
#                            args.number_of_classes,
#                            is_training=is_training,
#                            dropout_keep_prob=0.5,
#                            global_pool=False,
#                            scope=args.vgg_model,
#                            spatial_squeeze=False
#                            )
#
#
# image_path = "./test.jpg"  # 本地的测试图片
#
# image_raw = tf.gfile.GFile(image_path, 'rb').read()
# # 一定要tf.float()，否则会报错
# image_decoded = tf.to_float(tf.image.decode_jpeg(image_raw))
# image_decoded = tf.reshape(image_decoded, (224, 224,3))
# # 扩展图片的维度，从三维变成四维，符合Vgg19的输入接口
# image_expand_dim = tf.expand_dims(image_decoded, 0)
# print(image_expand_dim.get_shape())
# _, end_points = vgg.vgg_16(image_expand_dim)
# with tf.Session() as sess:
#     restorer = tf.train.Saver()
#     print("Start restore!")
#     restorer.restore(sess, "./pretrain/vgg16/" + "vgg_16" + ".ckpt")
#     print("Model checkpoits for " + "vgg16" + " restored!")


# 加载模型的参数
model_reader = pywrap_tensorflow.NewCheckpointReader(r"./pretrain/vgg16/vgg_16.ckpt")

# 然后，使reader将模型的参数变换称为dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()

# 最后，循环打印输出
for key in var_dict:
    print("variable name: ", key)
    print(model_reader.get_tensor(key))
    print(model_reader.get_tensor(key).shape)


