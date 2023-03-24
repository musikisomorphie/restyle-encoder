from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	},
	"cars_encode": {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['cars_train'],
		'train_target_root': dataset_paths['cars_train'],
		'test_source_root': dataset_paths['cars_test'],
		'test_target_root': dataset_paths['cars_test']
	},
	"church_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['church_train'],
		'train_target_root': dataset_paths['church_train'],
		'test_source_root': dataset_paths['church_test'],
		'test_target_root': dataset_paths['church_test']
	},
	"horse_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['horse_train'],
		'train_target_root': dataset_paths['horse_train'],
		'test_source_root': dataset_paths['horse_test'],
		'test_target_root': dataset_paths['horse_test']
	},
	"afhq_wild_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['afhq_wild_train'],
		'train_target_root': dataset_paths['afhq_wild_train'],
		'test_source_root': dataset_paths['afhq_wild_test'],
		'test_target_root': dataset_paths['afhq_wild_test']
	},
	"toonify": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	},
	'ham10k': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['ham10k'],
        'train_target_root': dataset_paths['ham10k'],
        'test_source_root': dataset_paths['ham10k'],
        'test_target_root': dataset_paths['ham10k']
    },
    'rxrx19b': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['rxrx19b'],
        'train_target_root': dataset_paths['rxrx19b'],
        'test_source_root': dataset_paths['rxrx19b'],
        'test_target_root': dataset_paths['rxrx19b']
    },
	'rxrx19b_HUVEC': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['rxrx19b_HUVEC'],
        'train_target_root': dataset_paths['rxrx19b_HUVEC'],
        'test_source_root': dataset_paths['rxrx19b_HUVEC'],
        'test_target_root': dataset_paths['rxrx19b_HUVEC']
    },
	'rxrx19a_HRCE': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['rxrx19a_HRCE'],
        'train_target_root': dataset_paths['rxrx19a_HRCE'],
        'test_source_root': dataset_paths['rxrx19a_HRCE'],
        'test_target_root': dataset_paths['rxrx19a_HRCE']
    },
	'rxrx19a_VERO': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['rxrx19a_VERO'],
        'train_target_root': dataset_paths['rxrx19a_VERO'],
        'test_source_root': dataset_paths['rxrx19a_VERO'],
        'test_target_root': dataset_paths['rxrx19a_VERO']
    },
	'CosMx': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['CosMx'],
        'train_target_root': dataset_paths['CosMx'],
        'test_source_root': dataset_paths['CosMx'],
        'test_target_root': dataset_paths['CosMx']
    },
	'Xenium': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['Xenium'],
        'train_target_root': dataset_paths['Xenium'],
        'test_source_root': dataset_paths['Xenium'],
        'test_target_root': dataset_paths['Xenium']
    },
    'Visium': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['Visium'],
        'train_target_root': dataset_paths['Visium'],
        'test_source_root': dataset_paths['Visium'],
        'test_target_root': dataset_paths['Visium']
    }
}