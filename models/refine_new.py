import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from datetime import datetime
from fuzzywuzzy import fuzz
import featuretools as ft
from memory_profiler import profile
import gc
import dask.dataframe as dd
import swifter  # for faster pandas operations
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

class DataQualityEnhancer:
    def __init__(self, memory_efficient=True, chunk_size=10000):
       
        self.column_types = {}
        self.categorical_encoders = {}
        self.feature_defs = None
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.scalers = {}
        
    def optimize_dtypes(self, df):
       
        optimized_df = df.copy()
        
        for column in optimized_df.columns:
            # Optimize integers
            if optimized_df[column].dtype.kind == 'i':
                if optimized_df[column].min() >= 0:
                    if optimized_df[column].max() < 255:
                        optimized_df[column] = optimized_df[column].astype(np.uint8)
                    elif optimized_df[column].max() < 65535:
                        optimized_df[column] = optimized_df[column].astype(np.uint16)
                    else:
                        optimized_df[column] = optimized_df[column].astype(np.uint32)
                else:
                    if optimized_df[column].min() > -128 and optimized_df[column].max() < 127:
                        optimized_df[column] = optimized_df[column].astype(np.int8)
                    elif optimized_df[column].min() > -32768 and optimized_df[column].max() < 32767:
                        optimized_df[column] = optimized_df[column].astype(np.int16)
                    else:
                        optimized_df[column] = optimized_df[column].astype(np.int32)
                        
            # Optimize floats
            elif optimized_df[column].dtype.kind == 'f':
                optimized_df[column] = optimized_df[column].astype(np.float32)
                
            # Optimize objects/strings
            elif optimized_df[column].dtype == 'object':
                if optimized_df[column].nunique() / len(optimized_df) < 0.5:  # If cardinality is low
                    optimized_df[column] = optimized_df[column].astype('category')
                    
        return optimized_df

    
    def _is_structured_text(self, series):
        """Check if text follows specific patterns (email, phone, etc.)"""
        # Sample a few non-null values
        sample = series.dropna().head(100)
        
        # Common patterns
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        phone_pattern = r'^\+?[\d\-\(\) ]{8,}$'
        url_pattern = r'^https?://[\w\.-]+\.\w+'
        
        if sample.empty:
            return False
            
        # Check if most values match any pattern
        pattern_matches = sample.astype(str).str.match(email_pattern) | \
                         sample.astype(str).str.match(phone_pattern) | \
                         sample.astype(str).str.match(url_pattern)
        
        return pattern_matches.mean() > 0.8

    def handle_missing_data(self, df):
        
        df_clean = df.copy()
        
        for column in df_clean.columns:
            # Ensure column type detection
            if column not in self.column_types:
                if pd.api.types.is_numeric_dtype(df_clean[column]):
                    self.column_types[column] = 'numerical'
                elif pd.api.types.is_string_dtype(df_clean[column]):
                    self.column_types[column] = 'categorical'
                elif pd.api.types.is_datetime64_any_dtype(df_clean[column]):
                    self.column_types[column] = 'datetime'
                else:
                    self.column_types[column] = 'text'
            
            # Handle missing values
            if df_clean[column].isnull().any():
                try:
                    if self.column_types[column] == 'numerical':
                        # Skewness-based strategy
                        if abs(df_clean[column].skew()) > 1:
                            imputer = SimpleImputer(strategy='median')
                        else:
                            imputer = SimpleImputer(strategy='median')
                        df_clean[column] = imputer.fit_transform(df_clean[[column]])
                    
                    elif self.column_types[column] == 'categorical':
                        # Cardinality-based strategy
                        if df_clean[column].nunique() > 10:
                            df_clean[column] = df_clean[column].fillna('Missing')
                        else:
                            df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
                    
                    elif self.column_types[column] == 'datetime':
                        # Custom datetime handling
                        df_clean = self._impute_datetime(df_clean, column)
                    
                    else:
                        # Default for text or undefined types
                        df_clean[column] = df_clean[column].fillna('Unknown')
                
                except Exception as e:
                    print(f"Warning: Issue with column '{column}'. Applying fallback. Error: {e}")
                    # Fallback to basic fillna
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                    else:
                        df_clean[column] = df_clean[column].fillna('Unknown')
        
        return df_clean
    
    def _impute_datetime(self, df, column):
        
        df_imputed = df.copy()
        
        # Convert to datetime if not already
        df_imputed[column] = pd.to_datetime(df_imputed[column], errors='coerce')
        
        # Sort by index or another relevant column
        df_sorted = df_imputed.sort_index()
        
        # Forward fill with a limit
        df_imputed[column] = df_sorted[column].fillna(method='ffill', limit=3)
        
        # Backward fill remaining
        df_imputed[column] = df_imputed[column].fillna(method='bfill')
        
        # Any remaining NaTs get the median date
        median_date = df_imputed[column].median()
        df_imputed[column] = df_imputed[column].fillna(median_date)
        
        return df_imputed

    def perform_dfs(self, df, primitives_config=None):
        """
        Memory-efficient Deep Feature Synthesis with null handling
        """
        # Get feature matrix through existing DFS process
        feature_matrix = self._chunked_dfs(df, primitives_config) if self.memory_efficient and len(df) > self.chunk_size else self._regular_dfs(df, primitives_config)
        
        # Handle nulls in generated features
        for column in feature_matrix.columns:
            if feature_matrix[column].isnull().any():
                # If numeric column
                if np.issubdtype(feature_matrix[column].dtype, np.number):
                    # If high skewness, use median
                    if abs(feature_matrix[column].skew()) > 1:
                        feature_matrix[column] = feature_matrix[column].fillna(feature_matrix[column].median())
                    # If normal distribution, use mean
                    else:
                        feature_matrix[column] = feature_matrix[column].fillna(feature_matrix[column].mean())
                # If categorical/object column
                else:
                    # If high cardinality, use 'Missing'
                    if feature_matrix[column].nunique() > 10:
                        feature_matrix[column] = feature_matrix[column].fillna('Missing')
                    # If low cardinality, use mode
                    else:
                        feature_matrix[column] = feature_matrix[column].fillna(feature_matrix[column].mode()[0])
        
        return feature_matrix
    
    def _chunked_dfs(self, df, primitives_config):
        """Process DFS in chunks for memory efficiency"""
        chunks = np.array_split(df, max(1, len(df) // self.chunk_size))
        feature_matrices = []
        
        for chunk in tqdm(chunks, desc="Processing DFS chunks"):
            fm = self._regular_dfs(chunk, primitives_config)
            feature_matrices.append(fm)
            gc.collect()  # Force garbage collection
            
        return pd.concat(feature_matrices, axis=0)
    
    def _regular_dfs(self, df, primitives_config):
        """Regular DFS processing"""
        # Reset index to ensure we have an index column
        df = df.reset_index(drop=True)
        
        # Define primitive configurations
        if primitives_config is None:
            primitives_config = {
                'agg_primitives': [
                    "mean", "sum", "std", "max", "min", "count",
                    "percent_true", "mode"
                ],
                'trans_primitives': [
                    "year", "month", "day",
                    "cum_sum", "diff"
                ]
            }
        
        # Create an EntitySet
        es = ft.EntitySet("dataset")
        
        # Add the main dataframe as an entity
        es = es.add_dataframe(
            dataframe_name="data",
            dataframe=df,
            index="index"
        )
        
        # Generate features
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="data",
            agg_primitives=primitives_config['agg_primitives'],
            trans_primitives=primitives_config['trans_primitives'],
            max_depth=2,
            verbose=True
        )
        
        # Store feature definitions
        self.feature_defs = feature_defs
        
        return feature_matrix
    
    def remove_correlated_features(self, df, threshold=0.95):
        """
        Remove highly correlated features from the dataset
        
        """
        # Only consider numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return df
            
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr(method="pearson").abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Drop features
        df_uncorrelated = df.drop(columns=to_drop)
        
        print(f"Removed {len(to_drop)} correlated features")
        return df_uncorrelated
        
    def standardize_and_encode(self, df):
        """
        Standardize numerical features and encode categorical ones
        """
        df_clean = df.copy()
        
        # Standardize numerical features
        numerical_cols = [col for col in df_clean.columns 
                         if self.column_types.get(col) == 'numerical']
        
        if numerical_cols:
            scaler = StandardScaler()
            df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])
        
        # Encode categorical features
        categorical_cols = [col for col in df_clean.columns 
                           if self.column_types.get(col) in ['categorical', 'text']]
        
        for col in categorical_cols:
            if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                known_classes = df_clean[col].dropna().unique()
                encoder = LabelEncoder()
                encoder.fit(known_classes)
                df_clean[col] = df_clean[col].map(
                    lambda x: encoder.transform([x])[0] if x in known_classes else -1
                )
                self.categorical_encoders[col] = encoder
        
        return df_clean

    def detect_column_types(self, df):
        """
        Enhanced column type detection with column dropping
        """
        drop_columns = []
        
        for column in df.columns:
            # Sample data for large datasets
            if len(df) > 10000:
                sample = df[column].sample(n=10000, random_state=42)
            else:
                sample = df[column]
                
            uniqueness_ratio = sample.nunique() / len(sample)
            null_ratio = sample.isnull().sum() / len(sample)
            
            # Drop columns with excessive nulls
            if null_ratio > 0.9:
                drop_columns.append(column)
                self.column_types[column] = 'drop_candidate'
                continue
            
            # Drop ID-like columns with very high uniqueness
            if uniqueness_ratio > 0.9:
                drop_columns.append(column)
                self.column_types[column] = 'id_drop_candidate'
                continue
        
        # Actually drop the identified columns
        if drop_columns:
            df.drop(columns=drop_columns, inplace=True)
            print(f"Dropped {len(drop_columns)} columns: {drop_columns}")
        
        return self.column_types

    def enhance_data(self, df, primitives_config=None):
        """
        Main pipeline with additional safeguards and optimizations
        """
        try:
            print("Detecting and dropping unnecessary columns...")
            self.detect_column_types(df)
           
            print("Optimizing data types...")
            df_optimized = self.optimize_dtypes(df)
            
            
            print("Detecting column types...")
            self.detect_column_types(df_optimized)
            
         
            print("Handling missing data...")
            df_clean = self.handle_missing_data(df_optimized)
            
           
            print("Performing Deep Feature Synthesis...")
            df_features = self.perform_dfs(df_clean, primitives_config)
            
            
            print("Removing correlated features...")
            df_uncorrelated = self.remove_correlated_features(df_features)
            
           
            print("Performing final preprocessing...")
            df_final = self.standardize_and_encode(df_uncorrelated)
            
            
            if self.memory_efficient:
                gc.collect()
            
            return df_final
            
        except Exception as e:
            print(f"Error in data enhancement pipeline: {str(e)}")
            raise


def process_large_dataset(file_path, chunk_size=10000):
    """
    Process a large dataset in chunks
    """
    # Read data in chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    enhancer = DataQualityEnhancer(memory_efficient=True, chunk_size=chunk_size)
    
    results = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        enhanced_chunk = enhancer.enhance_data(chunk)
        results.append(enhanced_chunk)
        
    return pd.concat(results, axis=0)

def handle_file(file):
    LARGE_FILE_SIZE = 30 * 1024 * 1024  # 10 MB in bytes
    file_size = os.path.getsize(file)

    if file_size > LARGE_FILE_SIZE:
        return process_large_dataset(file)
    else:
        df = pd.read_csv(file)
        enhancer = DataQualityEnhancer(memory_efficient=False)
        enhanced_df = enhancer.enhance_data(df)
        print("Processing small dataset...")
        return enhanced_df
    
enhanced=handle_file('trainames.csv')
enhanced.to_csv('output_enhanced.csv', index=False)