import tensorflow.keras as K
import auxilary
import losses

# ============================================================================================
# MANUAL PARAMETER SETTINGs
def main_train():
    learn_rate = 0.001
    nEpochs = 700
    nTrainImages = 64
    nValidImages = 32
    batchsize = 12
    stepsPerEpoch = nTrainImages/batchsize
    val_steps = nValidImages/batchsize

    inputPathImg = 'input path to training images (labelled) '
    inputPathLabel = 'input path to training labels' # see the generator for information on folder structure inside the label-folder
    inputPathImgUnlabeled = 'input path to training images (unlabelled)'

    inputPathImgValid = 'input path to validation images (labelled)'
    inputPathLabelValid = 'input path to validation labels'
    inputPathImgUnlabeledValid = 'input path to validation images (unlabelled)'

    input_depth = 3
    output_depthSeg = 1
    imgsize = (480, 480)
    regularisation = K.regularizers.l2(1e-5)

    # ============================================================================================
    # BUILD MODEL
    model = auxilary.ConsInstancy_Net(input_depth=input_depth, output_depthSeg=output_depthSeg, regularisation=regularisation, doBatchnorm=False)
    model.summary()

    train_generator = auxilary.generator_ConsInstancy(inputPathImg=inputPathImg,
                                                            inputPathLabel=inputPathLabel,
                                                            inputPathImgUnlabeled=inputPathImgUnlabeled,
                                                            in_out_size = imgsize,
                                                            batchsize=batchsize)
    validation_generator = auxilary.generator_ConsInstancy(inputPathImg=inputPathImgValid,
                                                            inputPathLabel=inputPathLabelValid,
                                                            inputPathImgUnlabeled=inputPathImgUnlabeledValid,
                                                            in_out_size = imgsize,
                                                            batchsize=batchsize)

    # ============================================================================================
    # TRAINING
    model.compile(
        optimizer=K.optimizers.Adam(lr=learn_rate),
        loss=[losses.semi_supervised_cosine_similarity_loss,
              losses.semi_supervised_Seg_loss,
              losses.semi_supervised_Seg_loss,
              losses.semi_supervised_Seg_loss,              
              losses.semi_supervised_dist_trafo_loss(model.get_layer('DistTrafo').output,
                                                     model.get_layer('DistTrafoInv').output, threshold=0.1)
              ],
        loss_weights=[1.0,
                      1.0,
                      1.0,
                      1.0,                      
                      1.0
                      ])

    history = model.fit(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=nEpochs,
        callbacks=[auxilary.checkpointer, auxilary.reduce_lr],
        validation_data=validation_generator,
        validation_steps=val_steps,
        verbose=1
        )
    

# ============================================================================================
if __name__ == "__main__":
    main_train()