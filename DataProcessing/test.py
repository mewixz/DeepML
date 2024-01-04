from DataProcessing import SuperCells
import numpy as np
SuperCells = SuperCells()
SuperCells.load(361108)
SuperCells.load(1)

data = SuperCells.tf_data_generator(361108, 'Train', 1)


from Models.models import SuperCellsModel


