# Sanity check for the latter function using kaf_cnn model
# Check that its prediction matches an adhoc model with weights loaded from the desired model
# Test result: WORKS
tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

kaf1_inp = get_hidden_layer_input(kaf_cnn, 'kaf_1', kaf_test_sample)

kaf_cnn_kaf1 = Sequential([
    layers.Conv2D(32, 3, padding='same', activation=None, kernel_initializer='he_uniform',  input_shape = (32, 32, 3), name = 'new_conv'),
    layers.BatchNormalization(name = 'new_bn') ])

# Set weights from kaf_cnn
wgts_conv = kaf_cnn.layers[0].get_weights()
kaf_cnn_kaf1.get_layer('new_conv').set_weights(wgts_conv)

wgts_bn = kaf_cnn.layers[1].get_weights()
kaf_cnn_kaf1.get_layer('new_bn').set_weights(wgts_bn)

do_match = tf.reduce_sum(tf.cast(tf.math.logical_not(tf.equal(kaf1_inp, kaf_cnn_kaf1.predict(kaf_test_sample))), dtype=tf.float32))

print (do_match) # 0.0 for True, False otherwise