import random
import traceback
import psutil

import tensorflow as tf
import numpy as np
from absl import logging


class DatasetBase(object):
    def __init__(self, data_dir, params):
        self._data_dir = data_dir
        self._split = params.split

        if "augmentations" in params:
            self._augmentations = params.augmentations
        else:
            self._augmentations = []

        # ----- Dataset Creation Params ----- #
        self._batch_size = params.batch_size

        self._cycle_length = psutil.cpu_count(logical=False)*2
        self._num_parallel_calls = psutil.cpu_count(logical=False)*2
        self._block_length = params.block_length
        self._buff_size = params.buff_size
        self._shuffle = params.shuffle

    def load(self, repeat=True):
        raise NotImplementedError

    def prepare(self, *args, **kwargs):
        raise NotImplementedError

    def _filter_collections(self, files):
        """Determines which stored value are returned and in what order"""

        return files

    def _apply_augmentations_generator(self, *args, **kwargs):
        # Store origional example once
        yield args

        for augmentation_set in self._augmentations:
            examples = [args]
            for augmentation in augmentation_set:
                augmented = []
                for example in examples:
                    augmented += augmentation(*example)
                examples = augmented

            # Do not add origional example for every augmentation step
            for i in range(len(augmented)):
                yield augmented[i]

    def _create_dataset_from_filepaths(self, files, repeat):
        try:
            npz = np.load(files[0], allow_pickle=True, encoding='latin1')
            npz_collections = self._filter_collections(npz.files)

            shapes = tuple(npz[key][0].shape for key in npz_collections)
            types = tuple(tf.as_dtype(npz[key].dtype) for key in npz_collections)
        except Exception as e:
            logging.error("%s file load unsuccessful from %s \n %s", type(self).__name__, files, str(e))
            logging.info(traceback.format_exc())
            raise e
        shard_dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = shard_dataset.interleave(lambda x: tf.data.Dataset.from_tensor_slices(tuple(zip(tf.py_func(func=self._make_generator,
                                                                                                             inp=(x, npz_collections),
                                                                                                             Tout=types)))),
                                           cycle_length=self._cycle_length,
                                           block_length=self._block_length)

        if self._augmentations:
            dataset = dataset.interleave(lambda *args: tf.data.Dataset.from_generator(lambda: self._apply_augmentations_generator(args),
                                                                                      output_types=types,
                                                                                      output_shapes=shapes),
                                         cycle_length=self._cycle_length)
            dataset = dataset.interleave(lambda x: tf.data.Dataset.from_tensor_slices(tuple(zip(tf.py_func(func=self._make_generator,
                                                                                                   inp=(x, npz_collections),
                                                                                                   Tout=types)))),
                                         cycle_length=self._cycle_length)

        # dataset = dataset.prefetch(2)

        # if self._shuffle:
        #     dataset = dataset.shuffle(self._buff_size * self._batch_size)
        #
        # dataset = dataset.batch(self._batch_size)
        #
        # if repeat:
        #     dataset = dataset.repeat()

        return dataset, shard_dataset

    def _make_generator(self, filename, npz_collections):
        try:
            npz = np.load(filename, allow_pickle=True, encoding='latin1')
        except FileNotFoundError as error:
            logging.fatal("Shard not found when producing generator fn: %s", filename)
            raise error

        collections = [npz[key.decode('utf-8')] for key in npz_collections]
        # for idx in range(collections[0].shape[0]):
        #     yield tuple(collection[idx] for collection in collections)
        return collections

    # def _create_episodic_dataset_from_nested_filespaths(self, episodes, shot, way, repeat):
    #     try:
    #         npz = np.load(episodes[0].split(",")[0], allow_pickle=True, encoding='latin1')
    #         npz_collections = self._filter_collections(npz.files)
    #
    #         # shapes = [npz[key][0].shape for key in npz_collections] * 2
    #         types = tuple([tf.as_dtype(npz[key].dtype) for key in npz_collections] * 2)
    #     except Exception as e:
    #         logging.error("%s file load unsuccessful from %s \n %s", type(self).__name__, episodes, str(e))
    #         logging.info(traceback.format_exc())
    #         raise e
    #     dataset = tf.data.Dataset.from_generator(self._make_episode_generator,
    #                                              args=(episodes, shot, way, npz_collections),
    #                                              output_types=types)
    #
    #     if self._augmentations:
    #         dataset = dataset.interleave(lambda *args: tf.data.Dataset.from_generator(self._apply_augmentations_generator,
    #                                                                                   args=args,
    #                                                                                   output_types=types),
    #                                      num_parallel_calls=self._num_parallel_calls,
    #                                      block_length=self._block_length,
    #                                      cycle_length=self._cycle_length)
    #
    #     if self._shuffle:
    #         dataset = dataset.shuffle(self._buff_size * self._batch_size)
    #     if repeat:
    #         dataset = dataset.repeat()
    #
    #     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #
    #     return dataset
    #
    # def _make_episode_generator(self, episodes, shot, way, npz_collections):
    #     for episode_classes_string in episodes:
    #         episode_classes_list = episode_classes_string.decode('utf-8').split(",")
    #         episode_classes = random.sample(list(episode_classes_list), way)
    #
    #         support = [[] for _ in range(len(npz_collections))]
    #         query = [[] for _ in range(len(npz_collections))]
    #         for class_file in episode_classes:
    #             try:
    #                 npz = np.load(class_file, allow_pickle=True, encoding='latin1')
    #             except FileNotFoundError as error:
    #                 logging.fatal("Shard not found when producing generator fn: %s", class_file)
    #                 raise error
    #
    #             collections = [npz[key.decode('utf-8')] for key in npz_collections]
    #
    #             sample_idxs = np.linspace(0., float(len(collections[0])-1), len(collections[0])).astype(np.int32)
    #             np.random.shuffle(sample_idxs)
    #
    #             for idx, collection in enumerate(collections):
    #                 support[idx].append(collection[sample_idxs[:shot]])
    #
    #             for idx, collection in enumerate(collections):
    #                 query[idx].append(collection[sample_idxs[shot:]])
    #
    #         yield tuple([np.concatenate(x, axis=0) for x in support] + [np.concatenate(x, axis=0) for x in query])

quickdraw_classes = "The Eiffel Tower,The Mona Lisa,aircraft carrier,alarm clock,ambulance,angel,animal migration,ant,apple,arm," \
                    "asparagus,banana,barn,baseball,baseball bat,bathtub,beach,bear,bed,bee,belt,bench,bicycle,binoculars,bird,blueberry," \
                    "book,boomerang,bottlecap,bread,bridge,broccoli,broom,bucket,bulldozer,bus,bush,butterfly,cactus,cake,calculator," \
                    "calendar,camel,camera,camouflage,campfire,candle,cannon,car,carrot,castle,cat,ceiling fan,cell phone,cello,chair," \
                    "chandelier,church,circle,clarinet,clock,coffee cup,computer,cookie,couch,cow,crayon,crocodile,crown,cruise ship," \
                    "diamond,dishwasher,diving board,dog,dolphin,donut,door,dragon,dresser,drill,drums,duck,dumbbell,ear,eye,eyeglasses," \
                    "face,fan,feather,fence,finger,fire hydrant,fireplace,firetruck,fish,flamingo,flashlight,flip flops,flower,foot,fork," \
                    "frog,frying pan,garden,garden hose,giraffe,goatee,grapes,grass,guitar,hamburger,hand,harp,hat,headphones,hedgehog," \
                    "helicopter,helmet,hockey puck,hockey stick,horse,hospital,hot air balloon,hot dog,hourglass,house,house plant," \
                    "ice cream,key,keyboard,knee,knife,ladder,lantern,leaf,leg,light bulb,lighter,lighthouse,lightning,line,lipstick," \
                    "lobster,mailbox,map,marker,matches,megaphone,mermaid,microphone,microwave,monkey,mosquito,motorbike,mountain,mouse," \
                    "moustache,mouth,mushroom,nail,necklace,nose,octopus,onion,oven,owl,paint can,paintbrush,palm tree,parachute," \
                    "passport,peanut,pear,pencil,penguin,piano,pickup truck,pig,pineapple,pliers,police car,pool,popsicle,postcard,purse," \
                    "rabbit,raccoon,radio,rain,rainbow,rake,remote control,rhinoceros,river,rollerskates,sailboat,sandwich,saxophone," \
                    "scissors,see saw,shark,sheep,shoe,shorts,shovel,sink,skull,sleeping bag,smiley face,snail,snake,snowflake," \
                    " ball,speedboat,square,star,steak,stereo,stitches,stop sign,strawberry,streetlight,string bean,submarine,sun," \
                    "swing set,syringe,t-shirt,table,teapot,teddy-bear,tennis racquet,tent,tiger,toe,tooth,toothpaste,tractor," \
                    "traffic light,train,triangle,trombone,truck,trumpet,umbrella,underwear,van,vase,watermelon,wheel,windmill," \
                    "wine bottle,wine glass,wristwatch,zigzag,blackberry,power outlet,peas,hot tub,toothbrush,skateboard,cloud,elbow," \
                    "bat,pond,compass,elephant,hurricane,jail,school bus,skyscraper,tornado,picture frame,lollipop,spoon,saw,cup," \
                    "roller coaster,pants,jacket,rifle,yoga,toilet,waterslide,axe,snowman,bracelet,basket,anvil,octagon,washing machine," \
                    "tree,television,bowtie,sweater,backpack,zebra,suitcase,stairs,The Great Wall of China".split(',')
quickdraw_lookup = {class_string: i for i, class_string in enumerate(sorted(quickdraw_classes))}
