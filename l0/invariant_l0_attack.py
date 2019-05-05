import tensorflow as tf
import random
import time
import numpy as np
from keras.datasets import mnist
import sys
import os
import itertools
import sklearn.cluster
import scipy.misc

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

DTYPE = tf.float32

def make_model(filters=64, s1=5, s2=5, s3=3,
               d1=0, d2=0, fc=256,
               lr=1e-3, decay=1e-3):
    model = Sequential()
    model.add(Conv2D(filters, kernel_size=(s1, s1),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters*2, (s2, s2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters*2, (s3, s3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(d1))
    model.add(Flatten())
    model.add(Dense(fc, activation='relu'))
    model.add(Dropout(d2))
    model.add(Dense(10))
    
    opt = keras.optimizers.Adam(lr, decay=decay)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    final = Sequential()
    final.add(model)
    final.add(Activation('softmax'))
    final.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
        
    
    return model, final

    
def train_model(model, x_train, y_train, batch_size=256,
                epochs=20):
    model.fit(x_train, keras.utils.to_categorical(y_train, 10),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=2,
    )

    return model


def show(img):
    img = img
    remap = " .*#" + "#" * 100
    img = (img.flatten()) * 3
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))

def compute_mat(angle, sx, sy, ax, ay, tx, ty, da, db):
    mat = np.eye(3)
    mat = np.dot(mat, [[1,ax,0],
                       [ay,1,0],
                       [0, 0, 1]])
    mat = np.dot(mat, [[sx,0,0],
                       [0,sy,0],
                       [0, 0, 1]])
    mat = np.dot(mat, [[1,0,tx],
                       [0,1,ty],
                       [0, 0, 1]])
    mat = np.dot(mat, [[np.cos(angle), np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
    
    inv = np.linalg.inv(mat)
    return mat, inv

def cluster(mask):
    dbscan = sklearn.cluster.DBSCAN(2, min_samples=5)
    points = [(i,j) for i in range(28) for j in range(28) if mask[0,i,j,0]]
    points = np.array(points)
    dbscan.fit(points)
    flat = points[:,0]*28+points[:,1]
    labels = dbscan.labels_
 
    arr = np.zeros((28*28))
    arr[flat] = -1
    
    for i in range(max(labels)+1):
        arr[flat[labels==i]] = 1+i
    arr = arr.reshape((28,28))
    return arr

def improve_transform():
    sys.path.append("gan/")
    from gan.acgan_mnist import Generator

    zin = tf.placeholder(tf.float32, [None, 74])
    x_target = tf.placeholder(tf.float32, [None, 28, 28, 1])

    generated_images, _ = Generator(None, zin)
    generated_images = tf.reshape(generated_images, [-1, 28, 28, 1])

    similarity_loss = tf.reduce_sum(np.abs(generated_images - x_target),axis=(1,2,3))
    z_loss = 0.01*tf.reduce_sum(zin[:,10:]**2, axis=1)
    total_loss = similarity_loss + z_loss
    grads = tf.gradients(similarity_loss, [zin])[0]

    sess = tf.Session()

    touse = [x for x in tf.trainable_variables() if 'Generator' in x.name]
    saver = tf.train.Saver(touse)
    saver.restore(sess, 'gan/model/mnist-acgan-2')
    
    keras.backend.set_learning_phase(False)
    
    def score(image, label):
        #show(image)
        zs = np.random.normal(0, 1, size=(128, 74))
        zs[:,:10] = 0
        zs[:,label] = 1
        
        for _ in range(30):
            #print("generate")
            ell, l_sim, l_z, nimg, delta = sess.run((total_loss, similarity_loss,
                                                     z_loss, generated_images,grads),
                                        {zin: zs,
                                         x_target: image[np.newaxis,:,:,:]})
            #print(l_sim)
            #show(nimg)
            zs[:,10:] -= delta[:,10:]*.01

        return np.min(ell)

    transformation_matrix = tf.placeholder(tf.float32, [8])
    xs = tf.placeholder(DTYPE, [None, 28, 28, 1])
    transformed = tf.contrib.image.transform(xs, transformation_matrix,
                                             'BILINEAR')

    uids = list(set([int(x.split("_")[1]) for x in os.listdir("best") if 'best_' in x and "_10000" in x]))

    num = [max([int(x.split("_")[2][:-4]) for x in os.listdir("best") if str(uids[i]) in x and 'idx' not in x and 'tran' not in x]) for i in range(4)]

    
    
    arr = []
    for fileid, filecount in zip(uids, num):
        best = np.load("best/best_%d_%d.npy"%(fileid,filecount))
        best_idx = np.array(np.load("best/best_%d_%d_idx.npy"%(fileid,filecount)), dtype=np.int32)
        best_transforms = np.load("best/best_%d_transforms_%d.npy"%(fileid,filecount))
        
        mask = (abs(best-x_test[use_idx]) > .5)
        delta = np.sum(mask,axis=(1,2,3))
        arr.append(delta)
        print(delta)
        print(np.median(delta))
    arr = np.min(arr,axis=0)
    
    fout = open("/tmp/out.html","w")
    
    def write(txt, img, lab, delta, doinv=False, do=True):
        if do:
            if len(img.shape) == 4:
                img = img[0]
            if doinv:
                timg = sess.run(transformed, {xs: img[np.newaxis,:,:,:],
                                             transformation_matrix: inv.flatten()[:-1]})[0]
            else:
                timg = img
                
            s = score(timg, lab)
        else:
            s = 0

        print(lab, type(lab))
        print(delta, type(delta))
        fout.write('<div style="float: left; padding: 3px">%d[%d]@%d<br/><img style="width:50px; height:50px;" src="%s"/></div>'%(int(s),lab,delta,txt))
        scipy.misc.imsave("/tmp/"+txt, img.reshape((28,28)))
        print("score of being", lab, "is:", s)
        show(img)
        fout.flush()
        return s
    
    candidates = []
    for IDX in range(100):
        fout.write("<br/><div style='clear: both'></div><br/>")
        mat, inv = compute_mat(*best_transforms[IDX])

        img = sess.run(transformed, {xs: x_train[best_idx[IDX:IDX+1]],
                                     transformation_matrix: mat.flatten()[:-1]})
        
        print("Source image")
        write("img_%d_0.png"%IDX, x_test[use_idx[IDX]], y_test[use_idx[IDX]],0)
        
        print("Target image")
        write("img_%d_2.png"%IDX, x_train[best_idx[IDX]], y_train[best_idx[IDX]],0)

        mask = (abs(x_test[use_idx[IDX]]-img) > .5)
        #origs.append(np.sum(mask))
        
        print("Transformed target image")
        write("img_%d_1.png"%IDX, img, y_train[best_idx[IDX]],np.sum(mask), True)

        write("img_%d_1.5.png"%IDX, np.array(mask,dtype=np.int32), y_train[best_idx[IDX]], np.sum(mask), True, do=False)
        
        print("Mask delta", np.sum(mask))
        show(mask)
        clusters = cluster(mask)
        print("\n".join(["".join([str(int(x)) for x in y]) for y in clusters]).replace("0"," ").replace("-1","*"))

        write("img_%d_1.6.png"%IDX, np.array(mask,dtype=np.int32), y_train[best_idx[IDX]], np.sum(mask), True, do=False)

        import matplotlib
        colored = np.zeros((28,28,3))
        for i in range(28):
            for j in range(28):
                if mask[0,i,j,0] != 0:
                    colored[i,j,:] = matplotlib.colors.to_rgb("C"+str(int(clusters[i,j]+1)))
        
        scipy.misc.imsave("/tmp/img_%d_1.6.png"%IDX, colored)

        possible = []
        
        for nid,subset in enumerate(itertools.product([False,True], repeat=int(np.max(clusters)))):
            if np.sum(subset) == 0: continue
            mask = np.any([clusters==(i+1) for i,x in enumerate(subset) if x], axis=0)+0.0
            mask = mask.reshape(img.shape)
            print("Mask weight", np.sum(mask))
            out = ((mask)*img) + ((1-mask)*x_test[use_idx[IDX]])
            print("New Image")
            s = write("img_%d_%d.png"%(IDX,3+nid), out, y_train[best_idx[IDX]], np.sum(mask), True)
            possible.append((out,s))
        candidates.append(possible)
            
        
            
        print("-"*80)

    import pickle
    pickle.dump(candidates, open("/tmp/candidates.p","wb"))

def find_transform():
    global x_train, x_test
    x_train = (x_train>.5) + 0
    x_test = (x_test>.5) + 0

    UID = random.randint(0,1000000)
    
    transformation_matrix = tf.placeholder(tf.float32, [8])
    inverse_matrix = tf.placeholder(tf.float32, [8])
    darkena = tf.placeholder(DTYPE, [])
    darkenb = tf.placeholder(DTYPE, [])

    print('shape',x_train.shape)
    dataset = tf.constant(x_train, dtype=DTYPE)
    labels = tf.constant(y_train, dtype=tf.int32)
    print('a1')
        
    transformed_dataset = tf.contrib.image.transform(dataset, transformation_matrix,
                                                     'BILINEAR')
    inverted_dataset = tf.contrib.image.transform(transformed_dataset, inverse_matrix,
                                                  'BILINEAR')
    ok_transform = tf.reduce_sum(inverted_dataset,axis=(1,2,3)) > tf.reduce_sum(dataset,axis=(1,2,3))*.85
    
    transformed_dataset = (1-(1-transformed_dataset)**darkenb)**(1./darkenb)
    print('a2')
    
    flat_transformed = tf.cast(tf.reshape(transformed_dataset, [-1, 28*28]), dtype=DTYPE)
    query = tf.placeholder(DTYPE, (None, 28, 28, 1))
    query_y = tf.placeholder(tf.int32, [None])
    
    query_t = tf.transpose(tf.reshape(query, [-1, 28*28]))
    query_t = (1-(1-query_t)**darkena)**(1./darkena)
    print('a3')
    
    norms = tf.reduce_sum(tf.square(flat_transformed), axis=1)[:, tf.newaxis] \
            - 2*tf.matmul(flat_transformed, query_t)
    
    badness1 = 1000*tf.reshape((1-tf.cast(ok_transform,dtype=DTYPE)),[-1,1])
    badness2 = 1000*tf.cast(tf.equal(tf.reshape(query_y, [1, -1]), tf.reshape(labels, [-1, 1])), dtype=DTYPE)
    print(norms, badness1, badness2, query_y, labels)
    norms = norms + badness1 + badness2
    _, topk_indices = tf.nn.top_k(-tf.transpose(norms), k=1, sorted=False)
    print('done')
    
    def rand(low,high):
        return random.random()*(high-low)+low
    
    sess = tf.Session()
    best = np.zeros((100,28,28,1))
    l0 = np.zeros(100)+10000
    best_idx = np.zeros(100)
    best_transforms = [None]*100
    
    for tick in range(10000000):
        angle = rand(-.25,.25)
        sx, sy = rand(.8,1.2), rand(.8,1.2)
        ax, ay = rand(-.2,.2), rand(-.2,.2)
        tx, ty = rand(-8,8), rand(-8,8)
        da, db = rand(-.25,4), rand(-.25,4)

        mat, inv = compute_mat(angle, sx, sy, ax, ay, tx, ty, da, db)
    
        now = time.time()
        ns, topk, dat, is_ok = sess.run((norms, topk_indices, transformed_dataset, ok_transform),
                                  {transformation_matrix: mat.flatten()[:-1],
                                   inverse_matrix: inv.flatten()[:-1],
                                   query: x_test[use_idx],
                                   query_y: y_test[use_idx],
                                   darkena: db,
                                   darkenb: db})
        #print(time.time()-now)

        for i in range(100):
            e = topk[i][0]
            v = ns[e, i]
    
            dd = np.sum((x_test[use_idx[i]]>.5)^(dat[e]>.5))
            #print('check', 'idx',i, 'to',e, 'val',v, 'was',best[i])
            if dd < l0[i]:
                #print("new better", 'idx',i, 'map to',e, 'was', best[i], 'now', v)
                #print('l0 diff',np.sum((x_train[i]>.5)^(dat[e]>.5)))
                l0[i] = min(l0[i], dd)
                best[i] = dat[e]
                best_idx[i] = e
                best_transforms[i] = [angle, sx, sy, ax ,ay, tx, ty, da, db]
        if tick%1000 == 0:
            print('mean',np.mean(l0),'median',np.median(l0))
            print(sorted(l0))
            np.save("best/best_%d_%d.npy"%(UID,tick),best)
            np.save("best/best_%d_%d_idx.npy"%(UID,tick),best_idx)
            np.save("best/best_%d_transforms_%d.npy"%(UID,tick),best_transforms)
        if tick%10000 == 0:
            for i in range(100):
                print("is",l0[i])
                show(x_test[use_idx[i]])
                show(best[i])
                show((x_test[use_idx[i]]>.5)^(best[i]>.5))    
    
x_train = y_train = None

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    img_rows = img_cols = 28
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #data_scores = np.load("mnist_scores.npy")
    #x_train = x_train[data_scores>1.0] # only keep the best 80% of the data
    #y_train = y_train[data_scores>1.0] # only keep the best 80% of the data
    use_idx = [159, 235, 247, 452, 651, 828, 937, 1018, 1021, 1543, 1567, 1692, 1899, 1904, 1930, 1944, 2027, 2082, 2084, 
               2232, 2273, 2306, 2635, 2733, 2805, 2822, 3169, 3290, 3335, 3364, 3394, 3469, 3471, 3540, 3628, 3735, 3999, 
               4014, 4086, 4329, 4456, 4471, 4482, 4496, 4503, 4504, 4611, 4630, 4649, 4726, 4840, 4974, 4980, 5089, 5209, 
               5281, 5447, 5522, 5700, 5820, 5909, 5926, 5946, 5988, 6054, 6130, 6408, 6506, 6558, 6693, 6759, 6762, 6779, 
               6881, 6947, 6997, 7031, 7063, 7154, 7377, 7547, 7625, 7759, 7790, 7796, 7826, 8334, 8535, 9073, 9181, 9195, 
               9231, 9375, 9458, 9563, 9639, 9695, 9720, 9811, 9825]
    
    #model, final = make_model()
    #train_model(final, x_train, y_train)
    #model.save("baseline.model")

    find_transform()
    #improve_transform()

