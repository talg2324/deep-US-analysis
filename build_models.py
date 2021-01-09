from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, Dropout, BatchNormalization, Input, Add, Cropping3D

def get_model():

    widths = {
                '1':4,
                '2':8,
                '3':16,
                'bottleneck':32
            }

    x0 = Input(shape=(1000,353,253,1))

    # -- Downsample layers --

    # Block 1
    xd11 = Conv3D(widths['1'], 3 , activation='relu')(x0)
    xd11 = BatchNormalization()(xd11)

    xd12 = Conv3D(widths['1'], 3, activation='relu')(xd11)
    xd12 = BatchNormalization()(xd12)

    # Block 2
    xd20 = MaxPooling3D(2)(xd12)
    
    xd21 = Conv3D(widths['2'], 3, activation='relu')(xd20)
    xd21 = BatchNormalization()(xd21)

    xd22 = Conv3D(widths['2'], 3, activation='relu')(xd21)
    xd22 = BatchNormalization()(xd22)

    # Block 3
    xd30 = MaxPooling3D(2)(xd22)
    
    xd31 = Conv3D(widths['3'], 3, activation='relu')(xd30)
    xd31 = BatchNormalization()(xd31)

    xd32 = Conv3D(widths['3'], 3, activation='relu')(xd31)
    xd32 = BatchNormalization()(xd32)

    # -- Bottleneck --

    xb = MaxPooling3D(2)(xd32)

    xb1 = Conv3D(widths['bottleneck'], 3, activation='relu')(xb)
    xb2 = Conv3D(widths['bottleneck'], 3, activation='relu')(xb1)

    # -- Upsample Layers --    
    xu30 = Conv3DTranspose(1, 2, strides=2, activation='relu')(xb2)    
    crop_dims = calc_cropping_layer(xu30, xd32)
    skip_connection = Cropping3D(crop_dims)(xd32)
    xu30 = Add()([xu30, skip_connection])

    xu31 = Conv3D(widths['3'], 3, activation='relu')(xu30)
    xu31 = BatchNormalization()(xu31)

    xu32 = Conv3D(widths['3'], 3, activation='relu')(xu31)
    xu32 = BatchNormalization()(xu32)

    # Block 2

    xu20 = Conv3DTranspose(1, 2, strides=2 , activation='relu')(xu32)
    crop_dims = calc_cropping_layer(xu20,xd22)
    skip_connection = Cropping3D(crop_dims)(xd22)
    xu20 = Add()([xu20, skip_connection])

    xu21 = Conv3D(widths['2'], 3, activation='relu')(xu20)
    xu21 = BatchNormalization()(xu21)

    xu22 = Conv3D(widths['2'], 3, activation='relu')(xu21)
    xu22 = BatchNormalization()(xu22)

    # Block 1
    xu10 = Conv3DTranspose(1, 2, strides=2 , activation='relu')(xu22)

    crop_dims = calc_cropping_layer(xu10,xd12)
    skip_connection = Cropping3D(crop_dims)(xd12)
    xu10 = Add()([xu10, skip_connection])

    xu11 = Conv3D(widths['1'], 3, activation='relu')(xu10)
    xu11 = BatchNormalization()(xu11)

    xu12 = Conv3D(widths['1'], 3, activation='relu')(xu11)
    xu12 = BatchNormalization()(xu12)

    model = Model(x0, xu12)
    model.summary()

    print('1')

def calc_cropping_layer(upsample, downsample):
    # Downsample layer is always bigger
    shape_mismatch = tuple(downsample.shape[i]-upsample.shape[i] for i in range(1,4))
    return tuple((i//2, i-i//2) for i in shape_mismatch)


if __name__ == "__main__":
    get_model()