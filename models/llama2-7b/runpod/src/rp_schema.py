INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0.1,
    },
    'top_p': {
        'type': float,
        'required': False,
        'default': 0.75,
    },
    'top_k': {
        'type': int,
        'required': False,
        'default': 40,
    },
    'num_beams': {
        'type': int,
        'required': False,
        'default': 1,
    },
    'max_length': {
        'type': int,
        'required': False,
        'default': 512,
    },
}
