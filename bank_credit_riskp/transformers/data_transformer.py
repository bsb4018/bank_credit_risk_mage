import pandas as pd
from sklearn.impute import SimpleImputer


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    df = df.rename(columns={'laufkont': 'status', 'laufzeit':'duration','moral': 'credit_history', 'verw':'purpose',\
                                    'hoehe': 'amount', 'sparkont':'savings','beszeit': 'employment_duration', 'rate':'installment_rate',\
                                        'famges': 'personal_status_sex', 'buerge':'other_debtors','wohnzeit': 'present_residence', \
                                            'verm':'property','alter': 'age', 'weitkred':'other_installment_plans',\
                                                'wohn': 'housing', 'bishkred':'number_credits','beruf': 'job', 'pers':'people_liable',\
                                                    'telef': 'telephone','gastarb':'foreign_worker','kredit': 'credit_risk'})
    
    simple_imputer_obj = SimpleImputer(strategy='most_frequent')
    simple_imputer_obj_fitted = simple_imputer_obj.fit(df)

    idf = pd.DataFrame(simple_imputer_obj_fitted.transform(df))
    idf.columns=df.columns
    idf.index=df.index
    
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
