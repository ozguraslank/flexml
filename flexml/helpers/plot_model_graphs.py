import plotly.graph_objects as go
import numpy as np
import shap
from typing import Union, Optional, Dict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from yellowbrick.regressor import ResidualsPlot, PredictionError


def plot_feature_importance(
        model: object,
        feature_names: list[str],
        top_x_features: int = 20,
        width: int = 800,
        height: int = 600,
    ) -> Union[go.Figure, str]:
    """
    Create a plotly figure showing feature importance for a given model

    Parameters
    ----------
    model: object
        Machine learning model to display its feature importance
    
    feature_names: list[str]
        List of feature names to display in the plot

    top_x_features: int (default = 20), optional
        Number of top features to display in the plot

    width: int (default = 800), optional
        Width of the plot   

    height: int (default = 600), optional
        Height of the plot

    Returns
    -------
    plotly.graph_objects.Figure or str
        A plotly figure object containing the feature importance visualization,
        or an error message if an error occurs during the process.
    """
    try:
        model_name = model.__class__.__name__
        importance = None

        # Check if the model has 'feature_importances_' attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_

        # Check if the model has coefficients (linear models)
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if importance.ndim > 1:  # Handle multi-output models (e.g., LogisticRegression with multiple classes)
                importance = np.mean(importance, axis=0)

        if importance is not None and len(importance) == len(feature_names):
            indices = np.argsort(importance)[::-1]  # Sort in descending order
            
            # Limit to top 20 features
            indices = indices[:top_x_features]
            sorted_importance = importance[indices]
            sorted_features = np.array(feature_names)[indices]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=sorted_features,
                x=sorted_importance,
                orientation='h',
                marker=dict(
                    color=sorted_importance,
                    colorscale='Viridis'
                )
            ))

            fig.update_layout(
                title=f"Feature Importance for {model_name} (Top {top_x_features} Features)",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=height,
                width=width,
                yaxis=dict(autorange="reversed")
            )

            return fig
        else:
            return f"Feature importance is not available or mismatched for {model_name}"

    except Exception as e:
        return f"Could not calculate feature importance for the model {model_name}. Error: {e}"


def plot_confusion_matrix(
        y_true: np.array, 
        y_pred: np.array, 
        class_mapping: dict = None,
        width: int = 800,
        height: int = 600
    ) -> Union[go.Figure, str]:
    """
    Create a plotly figure showing confusion matrix.

    Parameters
    ----------
    y_true : np.array
        Array of true (correct) labels

    y_pred : np.array
        Array of predicted labels

    class_mapping : dict, optional
        Dictionary mapping encoded values to class labels (e.g., {0: 'male', 1: 'female'})

    width: int (default = 800), optional
        Width of the plot

    height: int (default = 600), optional
        Height of the plot

    Returns
    -------
    plotly.graph_objects.Figure or str
        A plotly figure object containing the confusion matrix visualization,
        or an error message if an error occurs during the process.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        # Convert class indices to labels using the provided mapping
        class_names = [class_mapping[i] for i in range(cm.shape[0])] if class_mapping else list(range(cm.shape[0]))

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Viridis',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted label',
            yaxis_title='True label',
            yaxis=dict(autorange="reversed"),
            width=width,
            height=height
        )
        
        return fig
    except Exception as e:
        return f"Error creating confusion matrix plot: {str(e)}"


def plot_roc_curve(
        y_true: np.array, 
        y_prob: np.array, 
        class_names: list = None,
        width: int = 800,
        height: int = 600
    ) -> Union[go.Figure, str]:
    """
    Create a plotly figure showing ROC curve.

    Parameters
    ----------
    y_true : np.array
        Array of true (correct) labels

    y_prob : np.array
        Array of predicted probabilities

    class_names : list, optional
        List of class names for multiple classes

    width: int (default = 800), optional
        Width of the plot

    height: int (default = 600), optional
        Height of the plot

    Returns
    -------
    plotly.graph_objects.Figure or str
        A plotly figure object containing the ROC curve visualization,
        or an error message if an error occurs during the process.
    """
    try:
        fig = go.Figure()
        
        # Handle binary classification
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {auc_score:.3f})',
                mode='lines'
            ))
            
        # Handle multi-class
        else:
            if class_names is None:
                class_names = [f'Class {i}' for i in range(y_prob.shape[1])]
                
            for i in range(y_prob.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                auc_score = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'{class_names[i]} (AUC = {auc_score:.3f})',
                    mode='lines'
                ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=width,
            height=height,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        return f"Error creating ROC curve plot: {str(e)}"


def plot_calibration_curve(
        y_true: np.array, 
        y_prob: np.array, 
        class_mapping: Optional[Dict[int, str]] = None,
        n_bins: int = 10, 
        strategy: str = 'uniform',
        width: int = 800,
        height: int = 600,
    ) -> Union[go.Figure, str]:
    """
    Create a plotly figure showing probability calibration curve.

    Parameters
    ----------
    y_true : np.array
        True labels (binary or multiclass)

    y_prob : np.array
        Predicted probabilities (shape [n_samples, n_classes] for multiclass)

    n_bins : int (default = 10), optional
        Number of bins to discretize the [0, 1] interval

    strategy : {'uniform', 'quantile'} (default = 'uniform'), optional
        Strategy used to define the widths of the bins

    width: int (default = 800), optional
        Width of the plot

    height: int (default = 600), optional
        Height of the plot

    class_mapping: Dict[int, str] (default = None), optional
        Dictionary mapping class indices to class names

    Returns
    -------
    plotly.graph_objects.Figure or str
        A plotly figure object containing the calibration curve visualization,
        or an error message if an error occurs during the process.
    """
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.preprocessing import LabelBinarizer
        
        fig = go.Figure()
        
        # Handle binary classification
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
                
            prob_true, prob_pred = calibration_curve(y_true, y_prob, 
                                                    n_bins=n_bins, 
                                                    strategy=strategy)
            
            class_name = class_mapping.get(1, 'Positive Class') if class_mapping else 'Calibration Curve'
            fig.add_trace(go.Scatter(
                x=prob_pred,
                y=prob_true,
                name=class_name,
                mode='lines+markers',
                marker=dict(size=8)
            ))
            
        # Handle multiclass using one-vs-rest approach
        else:
            lb = LabelBinarizer().fit(y_true)
            y_onehot = lb.transform(y_true)
            
            for class_idx in range(y_prob.shape[1]):
                prob_true, prob_pred = calibration_curve(y_onehot[:, class_idx], 
                                                        y_prob[:, class_idx],
                                                        n_bins=n_bins,
                                                        strategy=strategy)
                
                class_name = class_mapping.get(class_idx, f'Class {class_idx}') if class_mapping else f'Class {class_idx}'
                
                # Apply class mapping here
                if class_mapping and class_idx in class_mapping:
                    class_name = class_mapping[class_idx]

                fig.add_trace(go.Scatter(
                    x=prob_pred,
                    y=prob_true,
                    name=class_name,
                    mode='lines+markers',
                    marker=dict(size=8)
                ))
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray'),
            mode='lines'
        ))
        
        fig.update_layout(
            title='Calibration Curve (Reliability Diagram)',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=width,
            height=height,
            showlegend=True,
            legend=dict(x=0.7, y=0.1),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
        
    except Exception as e:
        return f"Error creating calibration curve plot: {str(e)}"


def plot_shap(
        model: object, 
        X_test: np.array, 
        shap_type: str = 'shap_summary'
    ) -> Union[go.Figure, str]:
    """
    Create a plotly figure showing SHAP values visualization.

    Parameters
    ----------
    model : object
        Trained model

    X_test : np.array
        Feature data for explanation

    shap_type : str
        Type of SHAP plot to generate:
        - 'shap_summary': shap.summary_plot
        - 'shap_violin': shap.plots.violin

    Returns
    -------
    plotly.graph_objects.Figure or str
        A plotly figure object containing the SHAP values visualization,
        or an error message if an error occurs during the process.
    """
    try:
        # Check if model is a tree-based model
        model_type = str(type(model))
        
        tree_based_models = [
            "RandomForest", "GradientBoosting", "AdaBoost", 
            "HistGradientBoosting", "DecisionTree", "ExtraTrees",
            "XGB", "CatBoost", "LGBM"
        ]
        is_tree_based = any(model_name in model_type for model_name in tree_based_models)
        
        if is_tree_based:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.KernelExplainer(model.predict, X_test)
            shap_values = explainer.shap_values(X_test, silent=True)
        
        if len(shap_values.shape) == 3: # Models like DecisionTree, RandomForest return probabilities for each class, Let's downgrade to 2D array
            shap_values = shap_values[:, :, 1]
        # Convert SHAP values to appropriate format if needed
        if isinstance(shap_values, list) and shap_type != 'shap_dependence':
            shap_values = np.array(shap_values).mean(axis=0)
        
        # Generate the appropriate SHAP plot based on shap_type
        if shap_type == 'shap_summary':
            shap.summary_plot(shap_values, X_test)
        elif shap_type == 'shap_violin':
            shap.plots.violin(shap_values, X_test)
        else:
            return f"Invalid shap_type: {shap_type}"
            
        return True
    
    except Exception as e:
        return f"Error creating SHAP plot: {str(e)}"


def plot_residuals(
        model: object, 
        X_train: np.array, 
        y_train: np.array,
        X_test: np.array, 
        y_test: np.array
    ) -> object:
    """
    Create a residuals plot using Yellowbrick.

    Parameters
    ----------
    model : object
        Trained regressor

    X_train : np.array
        Training features

    y_train : np.array
        Training targets

    X_test : np.array
        Test features

    y_test : np.array
        Test targets

    Returns
    -------
    object
        Visualizer object from Yellowbrick
    """
    try:
        if model.__class__.__name__ == "CatBoostRegressor": # https://github.com/DistrictDataLabs/yellowbrick/issues/1099
            from yellowbrick.contrib.wrapper import regressor as wrap_regressor
            model = wrap_regressor(model)

        visualizer = ResidualsPlot(model)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        return visualizer
    
    except Exception as e:
        return f"Error creating residuals plot: {str(e)}"


def plot_prediction_error(
        model: object,
        X_train: np.array,
        y_train: np.array,
        X_test: np.array, 
        y_test: np.array
    ) -> object:
    """
    Create a prediction error plot using Yellowbrick.

    Parameters
    ----------
    model : object
        Trained regressor

    X_train : np.array
        Training features

    y_train : np.array
        Training targets

    X_test : np.array
        Test features

    y_test : np.array
        Test targets

    Returns
    -------
    object
        Visualizer object from Yellowbrick
    """
    try:
        if model.__class__.__name__ == "CatBoostRegressor": # https://github.com/DistrictDataLabs/yellowbrick/issues/1099
            from yellowbrick.contrib.wrapper import regressor as wrap_regressor
            model = wrap_regressor(model)

        visualizer = PredictionError(model)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        return visualizer
    
    except Exception as e:
        return f"Error creating prediction error plot: {str(e)}"