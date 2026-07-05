import os
import numpy as np
from PIL import Image

DATASET_PATH = "dataset_backup"
TEST_RATIO   = 0.2
LOAD_SEED    = 42

CLASS_NAMES = [
    'Alaf (a)', 'Ba (b)', 'Taa (t)', 'Tha (th)', 'Jeem (j)',
    'Haa (h)', 'Kha (kh)', 'Dal (d)', 'Thal (dh)', 'Raa (r)'
]
CLASS_NAMES_ASCII = [
    'Alaf  (0)', 'Ba    (1)', 'Taa   (2)', 'Tha   (3)', 'Jeem  (4)',
    'Haa   (5)', 'Kha   (6)', 'Dal   (7)', 'Thal  (8)', 'Raa   (9)'
]

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot

def augment_image_advanced(img):
    """
    Advanced stroke-preserving handwriting data augmentation:
    - Random scaling (85% to 115%)
    - Random shear slant (horizontal shear ±0.12)
    - Random rotation (±12 degrees)
    - Random translation (±2.0 pixels in x and y)
    - Stroke thickness adjustment (simulating varying pen pressures)
    """
    scale = np.random.uniform(0.85, 1.15)
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    
    if scale > 1.0:
        img_scaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img_scaled.crop((left, top, left + w, top + h))
    else:
        img_scaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        background = Image.new('L', (w, h), 0)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        background.paste(img_scaled, (left, top))
        img = background

    # 1. Random Shear (slanted handwriting)
    shear = np.random.uniform(-0.12, 0.12)
    img = img.transform(img.size, Image.Transform.AFFINE, 
                        (1, shear, 0, 0, 1, 0), 
                        resample=Image.Resampling.BILINEAR)

    # 2. Random rotation & translation
    angle = np.random.uniform(-12, 12)
    tx = np.random.uniform(-2.0, 2.0)
    ty = np.random.uniform(-2.0, 2.0)
    img = img.rotate(angle, translate=(tx, ty), 
                     resample=Image.Resampling.BILINEAR, fillcolor=0)

    # 3. Stroke thickness variation (Gaussian blur + Gamma/Contrast adjustment)
    if np.random.rand() > 0.5:
        from PIL import ImageFilter
        blur_radius = np.random.uniform(0.1, 0.4)
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        
        arr = np.array(img, dtype=np.float64)
        gamma = np.random.uniform(0.7, 1.3)
        arr = 255.0 * ((arr / 255.0) ** gamma)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        
    return img

def load_dataset(dataset_path=DATASET_PATH, test_ratio=TEST_RATIO, seed=LOAD_SEED, augment_factor=15):
    """
    Loads 50 original hand-drawn drawings per class from dataset_backup.
    Splits them strictly into 80% train originals (40 per class) and 20% test originals (10 per class).
    Augments the training originals in-memory by augment_factor to form a large train set (6,000 images).
    Keeps the test set as pure, completely unseen, original drawings (100 images) with zero leakage.
    """
    np.random.seed(seed)
    
    class_folders = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    
    X_train_list, Y_train_list = [], []
    X_test_list, Y_test_list = [], []
    
    print("=" * 60)
    print("Loading & Splitting Dataset (Disjoint Originals Split — Option A & B)")
    print("=" * 60)
    print(f"Found {len(class_folders)} class folders in {dataset_path}:")
    

    # checks if the files are images
    for idx, folder in enumerate(class_folders):
        folder_path = os.path.join(dataset_path, folder)
        file_names  = sorted([f for f in os.listdir(folder_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Shuffle files of this class before splitting [TO train the model of both handwritings]
        shuffled_files = np.random.permutation(file_names)
        n_test = int(len(shuffled_files) * test_ratio)
        test_files = shuffled_files[:n_test]
        train_files = shuffled_files[n_test:]
        
        print(f"  Class {idx} ({folder}): {len(file_names)} originals => {len(train_files)} train / {len(test_files)} test")
        
        # 1. Process Test Files (Keep as clean originals, only normalize)
        for fname in test_files:
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert('L')
                if img.size != (28, 28):
                    img = img.resize((28, 28))
                arr = np.array(img, dtype=np.float64) / 255.0
                X_test_list.append(arr.flatten())
                Y_test_list.append(idx)
            except Exception as e:
                print(f"    [Warning] Skipped test file {fname}: {e}")
                
        # 2. Process Train Files (Keep original + generate (augment_factor-1) variants)
        for fname in train_files:
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert('L')
                if img.size != (28, 28):
                    img = img.resize((28, 28))
                
                # Add original to train list
                arr = np.array(img, dtype=np.float64) / 255.0
                X_train_list.append(arr.flatten())
                Y_train_list.append(idx)
                
                # Generate augmented variants
                for _ in range(augment_factor - 1):
                    aug_img = augment_image_advanced(img)
                    arr_aug = np.array(aug_img, dtype=np.float64) / 255.0
                    X_train_list.append(arr_aug.flatten())
                    Y_train_list.append(idx)
            except Exception as e:
                print(f"    [Warning] Skipped train file {fname}: {e}")
                
    X_train = np.array(X_train_list, dtype=np.float64)
    X_test  = np.array(X_test_list, dtype=np.float64)
    
    Y_train_labels = np.array(Y_train_list, dtype=np.int32)
    Y_test_labels  = np.array(Y_test_list, dtype=np.int32)
    
    # One-hot encode labels
    Y_train = one_hot_encode(Y_train_labels, len(class_folders))
    Y_test  = one_hot_encode(Y_test_labels, len(class_folders))
    
    # Shuffle final train set so mini-batches are mixed
    train_idx = np.random.permutation(X_train.shape[0])
    X_train = X_train[train_idx]
    Y_train = Y_train[train_idx]
    
    # Shuffle final test set
    test_idx = np.random.permutation(X_test.shape[0])
    X_test = X_test[test_idx]
    Y_test = Y_test[test_idx]
    
    print(f"\nFinal Shapes:")
    print(f"  X_train: {X_train.shape}  |  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}  |  Y_test:  {Y_test.shape}\n")
    
    return X_train, Y_train, X_test, Y_test
