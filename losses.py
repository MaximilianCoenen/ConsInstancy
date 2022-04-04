import tensorflow.keras as K
import tensorflow as tf

# ====================================================================================================================================================
# pixelwise loss: positive and negative set of groundtruth pixels are weighted 50:50
def pixelwise_doubleMSELoss(y_true, y_pred, threshold=0.0):
    eps = 1e-5
    tThresh = tf.ones_like(y_true)*threshold

    wPositive = tf.cast(tf.math.greater(y_true, tThresh), 'float32')
    wNegative = tf.cast(tf.math.less_equal(y_true, tThresh), 'float32')    
    wSquare = tf.math.square(y_pred - y_true)
    
    nPositive = tf.math.reduce_sum(wPositive)
    conditionPos = tf.math.greater(nPositive, 0.0)
    sumPositive = K.backend.switch(conditionPos, tf.math.reduce_sum(wSquare * wPositive) / (nPositive+eps), 0.0)
    #sumPositive = K.sum(wSquare * wPositive) / nPositive    

    nNegative = tf.math.reduce_sum(wNegative)    
    conditionNeg = tf.math.greater(nNegative, 0.0)
    sumNegative = K.backend.switch(conditionNeg, tf.math.reduce_sum(wSquare * wNegative) / (nNegative+eps), 0.0)
    #sumNegative = K.sum(wSquare * wNegative) / nNegative    

    loss = sumPositive + sumNegative
    return loss




# ====================================================================================================================================================
def angular_cosine_similarity_loss(y_true, y_pred):
    errorAngles =  tf.reduce_sum(y_pred * y_true, axis=-1)
    loss = -1.0 * tf.reduce_mean(errorAngles)
    return loss


# ====================================================================================================================================================
def semi_supervised_Seg_loss(y_true, y_pred, threshold=0.0):
    b = tf.cast(tf.shape(y_true)[0], "int32")   
    b_half = tf.cast(b/2,"int32")
    
    loss = pixelwise_doubleMSELoss(y_true[b_half:b,], y_pred[b_half:b,], threshold=threshold)    
    return loss

# ====================================================================================================================================================
def semi_supervised_cosine_similarity_loss(y_true, y_pred):
    b = tf.cast(tf.shape(y_true)[0], "int32")   
    b_half = tf.cast(b/2,"int32")
    
    loss = angular_cosine_similarity_loss(y_true[b_half:b,], y_pred[b_half:b,])
    return loss



# ====================================================================================================================================================
def semi_supervised_dist_trafo_loss(dist_trafo, dist_trafo_inv, threshold=0.0):
    y_dist=tf.add(dist_trafo,dist_trafo_inv)
    def loss(y_true, y_pred):
        b = tf.cast(tf.shape(y_true)[0], "int32")
        b_half = tf.cast(b / 2, "int32")
        seg_main_ = y_pred[0:b_half, ]
        seg_dist_ = y_dist[0:b_half, ]
        
        # dist trafo consistency
        dist_trafo_consistency_loss = pixelwise_doubleMSELoss(seg_main_, seg_dist_, threshold=threshold)
        supervised_seg_loss = semi_supervised_Seg_loss(y_true, y_pred)
        return dist_trafo_consistency_loss+supervised_seg_loss

    return loss