import glob
import os
import re

clear_empty_urls = True

# get all training recips
recipes = sorted(glob.glob('configs/*/*.yaml'))
print('Total number of training recipes: ', len(recipes))
ar = glob.glob('configs/*/*_ascend.yaml')
print('Ascend training recipes: ', len(ar))
gr =  glob.glob('configs/*/*_gpu.yaml')
print('GPU training recipes: ', len(gr))
for item in (set(recipes) - set(ar) - set(gr)):
    print(item)

models_with_train_rec = []
for r in recipes:
    mn = r.split('/')[-2]
    if mn not in models_with_train_rec:
        models_with_train_rec.append(mn)
models_with_train_rec = sorted(models_with_train_rec)

print('\n==> Models with training recipes: ', len(models_with_train_rec))
print(models_with_train_rec)

# get readme file list
config_dirs = sorted([d for d in os.listdir('./configs') if os.path.isdir('configs/'+d)])
print('\nTotal number of config folders: ', len(config_dirs))
print('==> Configs w/o training rec: ', set(config_dirs)- set(models_with_train_rec))
readmes = [f'configs/{d}/README.md' for d in config_dirs]

for readme in readmes:
    if not os.path.exists(readme):
        print('Missing readme: ', readme)

# check yaml and reported performance 


# merge readme reported results
print('\r\n ')
output_path = './benchmark_results.md'
fout = open(output_path, 'w')

kw = ['Model', 'Top', 'Download', 'Config']

# process table head
head = '| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |'
fout.write(head + '\n')

fout.write("|----------------|------------|-------|-------|----------|-------|--------|---|-----|-----|" + '\n')

attrs = head.replace(' ','')[1:-1].split('|')
print('table attrs: ', attrs)

result_kw = ['Results', 'Benchmark', 'Result'] #TODO: unify this name
head_detect_kw = ['Model', 'Top', 'Config']

# process each model readme
parsed_models = []
parsed_model_specs  = []
for r in readmes:
    state = 0
    print('parsing ', r)
    results = []
    with open(r) as fp:
        for line in fp:
            if state==0:
                for kw in result_kw:
                    if f"##{kw}" in line.strip().replace(' ', ''):
                        state = 1
            # detect head
            elif state==1: 
                if '|Model|Context' in line.replace(' ', ''):
                    if len(line.split('|')) == len(head.split('|')):
                        state =2
                    else:
                        print('Detect head, but format is incorrect:')
                        print(line)
            
            # get table values
            elif state==2:
                if len(line.split('|')) == len(head.split('|')):
                    # try to parse each attr
                    attr_vals = line.replace(' ','')[1:-1].split('|')
                    download, config, log = attr_vals[-4:-1]
                    # clear empty urls
                    if clear_empty_urls:
                        if '.ckpt' not in download:
                            line = line.replace(download, "")
                        if ('.yaml' not in config) and ('.yml' not in config):
                            line = line.replace(config, "")
                        if '.log' not in log:
                            line = line.replace(log, "")
                    # clear empty model
                    if '--' not in line and re.search(r'[a-zA-Z]', attr_vals[0]):
                        results.append(line)
                        fout.write(line)
                        parsed_model_specs.append(line.split('|')[0])
                else:
                    state = 3
                    parsed_models.append(r.split('/')[1])
        
    if state==0:
        print('Fail to get Results')
    elif state==1:
        print('Fail to get table head')
    elif state==2:
        print('Fail to get table values')

print('Parsed models in benchmark: ', len(parsed_models))
print('Parsed model specs in benchmark: ', len(parsed_model_specs))
print('Readme using inconsistent result table format: \r\n', set(config_dirs) - set(parsed_models))


# write notes

fout.write('\n#### Notes\n')
fout.write('- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.\n- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.')

fout.close()

## check completeness: training recipes, recorded results, model py url
# 1) url check
#import mindcv

