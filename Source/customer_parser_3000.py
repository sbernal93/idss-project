from data import Data
from questions_generator import ask_questions, get_leaves


if __name__ == '__main__':
    data = Data()

    upsampled = data.upsampled_data_with_outliers
    dt_upsampled_x = data.upsampled_x_with_outliers
    downsampled = data.downsampled_data_with_outliers
    dt_downsampled_x = data.downsampled_x_with_outliers

    outliers_upsampled = data.upsampled_data_without_outliers
    dt_outliers_upsampled = data.upsampled_x_without_outliers
    outliers_downsampled = data.downsampled_data_without_outliers
    dt_outliers_downsampled = data.downsampled_x_without_outliers

    is_leaves = get_leaves(dt_upsampled_x)
    ask_questions(dt_upsampled_x, is_leaves, upsampled.drop('deposit', 1).columns.tolist())

    # is_leaves = get_leaves(dt_downsampled_x)
    # ask_questions(dt_downsampled_x, is_leaves, downsampled.drop('deposit', 1).columns.tolist())
