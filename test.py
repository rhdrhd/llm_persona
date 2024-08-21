from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
import parlai.utils.logging as logging

# Set up the argument parser
parser = ParlaiParser()
parser.set_params(
    task='personachat',
    datatype='train:ordered',  # 'train:ordered', 'valid', 'test'
    num_epochs=1,
)
opt = parser.parse_args(print_args=False)

# Create the task (without an agent)
world = create_task(opt)

# Iterate over the dataset
for _ in range(10):  # Adjust the range to see more examples
    world.parley()
    print(world.display())
    if world.epoch_done():
        break

# Reset logging
logging.disable()