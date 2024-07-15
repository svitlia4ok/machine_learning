from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Dict, Any

def getInputsAndTargets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Виділяє вхідні дані та цільові значення з DataFrame.

    Args:
        df (pd.DataFrame): Вхідний DataFrame з даними.

    Returns:
        dict: Словник з ключами 'inputs' і 'targets', що містять відповідні DataFrame.
    """
    input_cols = df.drop(columns=["Exited"]).columns
    target_col = ["Exited"]
    inputs = df[input_cols]
    targets = df[target_col]
    return {
        'inputs': inputs,
        'targets': targets
    }

def scaleInputs(scalerObj: BaseEstimator, inputs_train: pd.DataFrame, inputs_val: pd.DataFrame, number_cols_to_scale: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Масштабує числові колонки вхідних даних.

    Args:
        scalerObj (BaseEstimator): Об'єкт скейлера.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.
        number_cols_to_scale (list): Список числових колонок для масштабування.

    Returns:
        dict: Словник з ключами 'inputs_train' і 'inputs_val', що містять масштабовані дані.
    """
    scalerObj.fit(inputs_train[number_cols_to_scale])
    inputs_train[number_cols_to_scale] = scalerObj.transform(inputs_train[number_cols_to_scale])
    inputs_val[number_cols_to_scale] = scalerObj.transform(inputs_val[number_cols_to_scale])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val
    }

def encodeInputs(encoderObj: BaseEstimator, categorical_cols: List[str], inputs_train: pd.DataFrame, inputs_val: pd.DataFrame) -> Dict[str, Any]:
    """
    Кодує категоріальні колонки вхідних даних.

    Args:
        encoderObj (BaseEstimator): Об'єкт енкодера.
        categorical_cols (list): Список категоріальних колонок для кодування.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.

    Returns:
        dict: Словник з ключами 'inputs_train', 'inputs_val' і 'categories_encoded_cols', що містять закодовані дані та назви закодованих колонок.
    """
    encoderObj.fit(inputs_train[categorical_cols])
    categories_encoded_cols = encoderObj.get_feature_names_out().tolist()
    inputs_train[categories_encoded_cols] = encoderObj.transform(inputs_train[categorical_cols])
    inputs_val[categories_encoded_cols] = encoderObj.transform(inputs_val[categorical_cols])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'categories_encoded_cols': categories_encoded_cols
    }

def preprocess_data(df: pd.DataFrame, scaleNumeric: bool = True) -> Dict[str, Any]:
    """
    Попередньо обробляє дані, включаючи масштабування числових колонок і кодування категоріальних колонок.

    Args:
        df (pd.DataFrame): Вхідний DataFrame з даними.
        scaleNumeric (bool): Вказує, чи потрібно масштабувати числові колонки.

    Returns:
        dict: Словник з попередньо обробленими даними та об'єктами скейлера і енкодера.
    """
    it = getInputsAndTargets(df)
    inputs = it['inputs']
    targets = it['targets']
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=0.1, random_state=42, stratify=df['Exited'])
    number_cols = inputs.select_dtypes(include="number").columns.to_list()
    categorical_cols = inputs.select_dtypes(include="object").columns.to_list()
    categorical_cols.remove('Surname')
    number_cols_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    
    scalerObj = MinMaxScaler()
    if scaleNumeric:
        scaled_data = scaleInputs(scalerObj, inputs_train, inputs_val, number_cols_to_scale)
        inputs_train = scaled_data['inputs_train']
        inputs_val = scaled_data['inputs_val']
    
    encoderObj = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_data = encodeInputs(encoderObj, categorical_cols, inputs_train, inputs_val)
    inputs_train = encoded_data['inputs_train']
    inputs_val = encoded_data['inputs_val']
    categories_encoded_cols = encoded_data['categories_encoded_cols']
    
    inputs_train = inputs_train[number_cols + categories_encoded_cols]
    inputs_val = inputs_val[number_cols + categories_encoded_cols]
    
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'targets_train': targets_train,
        'targets_val': targets_val,
        'scalerObj': scalerObj,
        'encoderObj': encoderObj,
        'number_cols': number_cols,
        'categorical_cols': categorical_cols,
        'number_cols_to_scale': number_cols_to_scale
    }

def preprocess_new_data(
        df: pd.DataFrame, 
        scalerObj: BaseEstimator, 
        encoderObj: BaseEstimator, 
        number_cols: List[str], 
        number_cols_to_scale: List[str],
        categorical_cols: List[str],
        scaleNumeric: bool = True
    ) -> pd.DataFrame:
    """
    Попередньо обробляє нові дані з використанням існуючих об'єктів скейлера і енкодера.

    Args:
        df (pd.DataFrame): Вхідний DataFrame з новими даними.
        scalerObj (BaseEstimator): Існуючий об'єкт скейлера.
        encoderObj (BaseEstimator): Існуючий об'єкт енкодера.
        number_cols (list): Список числових колонок.
        number_cols_to_scale (list): Список числових колонок для масштабування.
        categorical_cols (list): Список категоріальних колонок для кодування.
        scaleNumeric (bool): Вказує, чи потрібно масштабувати числові колонки.

    Returns:
        pd.DataFrame: Попередньо оброблений DataFrame з новими даними.
    """
    if scaleNumeric:
        df[number_cols_to_scale] = scalerObj.transform(df[number_cols_to_scale])
    
    categories_encoded_cols = encoderObj.get_feature_names_out().tolist()
    df[categories_encoded_cols] = encoderObj.transform(df[categorical_cols])
    
    df = df[number_cols + categories_encoded_cols]
    
    return df