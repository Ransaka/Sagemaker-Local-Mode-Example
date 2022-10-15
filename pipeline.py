from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (ProcessingStep,ProcessingOutput,ProcessingInput,
                                      TrainingInput, TrainingStep)


from sagemaker.workflow.pipeline_context import LocalSession

DUMMY_IAM_ROLE = 'IAM_ROLE'
output_path_s3 = 'S3_PATH'
SESSION = LocalSession()
SESSION.config = {'local': {'local_code': True}}

def get_pipeline():
    processor = ScriptProcessor(
        command = ["python3"],
        image_uri = 'CONTAINER_IMG_PREPROCESS',
        role = DUMMY_IAM_ROLE,
        instance_count = 1,
        instance_type = "local"
    )

    step_data_preprocessing = ProcessingStep(
        name = 'preprocessing',
        processor=processor,
        code="preprocess/preprocess.py",
        inputs=[ProcessingInput(source = 'data',destination= '/opt/ml/processing/input/data')],
        outputs=[ProcessingOutput(source='/opt/ml/processing/preprocessed',destination=output_path_s3)]
    )

    estimator = Estimator(
                output_path= 'file://'+'model',
                image_uri= 'CONTAINER_IMG_TRAIN',
                entry_point='train/train.py',
                role=DUMMY_IAM_ROLE,
                sagemaker_session=SESSION,
                instance_count= 1,
                instance_type='local',
    )

    step_train_model = TrainingStep(
        name = 'training',
        estimator=estimator,
        inputs={
            f"train": TrainingInput(
                s3_data= output_path_s3,
                content_type="text/csv",
            )},
        depends_on=[step_data_preprocessing]
    )

    pipeline = Pipeline(
        name= 'local pipeline',
        steps=[
            step_data_preprocessing,
            step_train_model
            ],
        sagemaker_session= SESSION
    )

    return pipeline