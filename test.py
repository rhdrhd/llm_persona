from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.worlds import create_task

@register_script('display_data')
class DisplayData(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True, 'Display data from a task')
        parser.add_argument('-n', '--num-examples', default=10, type=int)
        parser.add_argument('-dt', '--datatype', default='train')
        return parser

    def run(self):
        opt = self.opt
        opt['task'] = 'personachat'
        world = create_task(opt, None)

        # Collect data
        data = []
        for _ in range(opt['num_examples']):
            world.parley()
            msg = world.get_acts()[0]
            data.append({
                'text': msg.get('text', ''),
                'labels': msg.get('labels', [''])[0],
                'episode_done': msg.get('episode_done', False)
            })

        return data

if __name__ == '__main__':
    script = DisplayData.main()
    import pandas as pd
    df = pd.DataFrame(script)
    print(df)
