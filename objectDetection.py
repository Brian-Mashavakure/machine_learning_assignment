class ObjectDetection(object):

    def __init__(self):
        self._objects = []

    def videoToFrames(self, video):
        print('...Splitting Video To Frames....')
        frameName = 'C:\\Users\\hp\\machine_learning_assignment\\static\\frames'
        vidCap = cv2.VideoCapture(video)
        success, image = vidCap.read()
        count = 0
        while vidCap.isOpened():
            success, frame = vidCap.read()
            if success:
                cv2.imwrite(frame_name + str(count) + '.jpg', frame)
                print(frame)
            else:
                break
            count = count + 1
        vidCap.release()
        cv2.destroyAllWindows()
        print('SPlitting Successful')

    def detect(self):
        print('feeding frames to inceptionV3...')
        for frame in self.get_frames():
            image = load_img(frame, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            y_pred = inceptionV3.predict(image)
            label = decode_predictions(y_pred)
            self._objects.append(label[0][1][1])
        print('Done feeding')
        objects_file = 'C:\\Users\\hp\\Desktop\\ins\\detectedObjects.txt'
        with open(objects_file, 'w') as f:
            f.write(json.dumps(self._objects))

    def get_frames(self):
        frames_arr = glob("C:\\Users\\hp\\Desktop\\ins\\frames\\*.jpg")
        return frames_arr

    def get_objects(self):
        return self._objects

    def search_objects(self, _object):
        print('searching...')
        objects_file = 'C:\\Users\\hp\\Desktop\\ins\\detectedObjects.txt'
        with open(objects_file, 'r') as objects_file:
            objects = list(json.loads(objects_file.read()))
        search_results = []
        if _object in set(objects):
            for index in range(len(objects)):
                if _object.__eq__(objects[index]):
                    img_url = self.get_frames()[index].split('/')[-1]
                    search_results.append(img_url)
        else:
            return 'Object was not found'
        return search_results

    def read_objects(self):
        objects_file = 'C:\\Users\\hp\\Desktop\\ins\\detectedObjects.txtt'
        with open(objects_file, 'r') as objects_file:
            objects = set(json.loads(objects_file.read()))
        return objects
