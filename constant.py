lpcv_class_names = ['Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Boat', 'Traffic Light',
                    'Stop Sign', 'Parking Meter', 'Bench', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant',
                    'Bear', 'Zebra', 'Backpack', 'Umbrella', 'Handbag', 'Tie', 'Skis', 'Sports Ball', 'Kite',
                    'Tennis Racket', 'Bottle', 'Wine Glass', 'Cup', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple',
                    'Orange', 'Broccoli', 'Hot Dog', 'Pizza', 'Donut', 'Chair', 'Couch', 'Potted Plant', 'Bed',
                    'Dining Table', 'Toilet', 'TV', 'Laptop', 'Mouse', 'Remote', 'Keyboard', 'Cell Phone', 'Microwave',
                    'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book', 'Clock', 'Vase', 'Teddy Bear', 'Hair Drier']

PROMPTS = [lambda name: f'itap of a {name}.',
           lambda name: f'a bad photo of the {name}.',
           lambda name: f'a origami {name}.',
           lambda name: f'a photo of the large {name}.',
           lambda name: f'a {name} in a video game.',
           lambda name: f'art of the {name}.',
           lambda name: f'a photo of the small {name}.']
