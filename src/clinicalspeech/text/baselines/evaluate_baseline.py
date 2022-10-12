import numpy as np
import pandas as pd
from clinicalspeech.utils import PROJECT_ROOT


class Scorer():
    ''' Class facilitating scoring baseline models '''
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def predict(self, 
                data: np.array, 
                ids: list, 
                labels: np.array, 
                split: str,
                binary: bool, 
                origin: str,
                model_id: str):
        ''' Predicts and saves prediction for a given model 
        Args:
            data (np.array): input data
            ids (list): examples ids for the inputs
            labels (np.array): true labels in numeric format
            split (str): train, val or test
            binary (bool): whether a binary or a multiclass model is trained
            origin (str): which positive class we are targeting (e.g., DEPR)
            model_id (str): a model identifier
            log_path (pathlib.Path): Path to save output file
        '''
        save_dir = PROJECT_ROOT / "model_predictions"
        save_file = save_dir / f'{model_id}_{split}.jsonl'
        predictions = self.model.predict_proba(data)
        pred_labels = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)
        if binary is False:
            label_mapping = self.config.multiclass_id2label_mapping 
        else:
            label_mapping = {0: 'TD', 1: origin}
        out_dict = {'trial_id': ids, 
                    'label': [label_mapping[p] for p in labels],
                    'origin': origin,
                    'id': ['_'.join(e.split('_')[:-1]) for e in ids],
                    'prediction': [label_mapping[p] for p in pred_labels],
                    'confidence': confidence,
                    'scores': predictions,
                    'model_name': model_id,
                    'binary': binary,
                    'type': 'text',
                    'split': split,
                    'target_class': origin,
                    'is_baseline': 1}
        out_df = pd.DataFrame(out_dict)
        out_df.to_json(save_file, orient='records', lines=True)

