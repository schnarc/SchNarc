import os
from schnarc.calculators import Queuer
import subprocess
import torch


class EnsembleTrainer:

    def __init__(self, ensemble_size, dataset_path, ensemble_dir, executable, input_template, basename='model'):
        self.ensemble_size = ensemble_size
        self.dataset_path = dataset_path
        self.ensemble_dir = ensemble_dir
        self.executable = executable
        self.basename = basename

        self.input_template = open(input_template, 'r').read()

        # Generate directory to hold ensemble models
        if not os.path.exists(self.ensemble_dir):
            os.makedirs(self.ensemble_dir)

        self.model_paths = []

    def train(self):
        raise NotImplementedError


class SchNarcTrainer(EnsembleTrainer):
    # TODO: First draft, needs to be adapted and a proper ensemble model

    def __init__(self, ensemble_size, dataset, ensemble_dir, executable, input_template, tradeoff_path, queuer=None,
                 basename='model'):
        super(SchNarcTrainer, self).__init__(ensemble_size, dataset, ensemble_dir, executable, input_template,
                                             basename=basename)

        self.tradeoff_path = tradeoff_path
        self.queuer = queuer
        self.step = 0

    def train(self):
        current_wdir = os.path.join(self.ensemble_dir, 'generation_{:06d}'.format(self.step))

        if not os.path.exists(current_wdir):
            os.makedirs(current_wdir)

        # Generate inputs
        input_files = self._write_inputs(current_wdir)

        if self.queuer is None:
            # Do sequential training
            for input_file in input_files:
                command = '{:s} with {:s}'.format(self.executable, input_file)
                training = subprocess.Popen(command, shell=True)
                training.wait()
        else:
            self.queuer.submit(input_files, current_wdir)

        self.step += 1

        # Finally, copy exported models to main ensemble directory for simulation
        self.model_paths = []
        for input_file in input_files:
            model_name = '{:s}.model'.format(os.path.splitext(input_file)[0])
            target_model = os.path.join(self.ensemble_dir, os.path.basename(model_name))
            subprocess.check_output('mv {:s} {:s}'.format(model_name, target_model), shell=True)
            self.model_paths.append(target_model)

    def update_ensemble(self, ensemble_calculator, device='cuda'):
        if self.model_paths is not None:
            models = [torch.load(model_path).to(device) for model_path in self.model_paths]
            ensemble_calculator.model = models

    def _write_inputs(self, current_wdir):
        input_files = []

        for i in range(self.ensemble_size):
            job_path = os.path.join(current_wdir, '{:s}_{:06d}'.format(self.basename, i + 1))
            export_path = '{:s}.model'.format(job_path)
            input_file = '{:s}.yaml'.format(job_path)

            db_path = self.dataset_path

            input_command = self.input_template.format(db_path=db_path, tradeoff_path=self.tradeoff_path,
                                                       job_path=job_path, export_path=export_path)
            with open(input_file, 'w') as inp:
                inp.write(input_command)

            input_files.append(input_file)

        return input_files


class SchNarcQueuer(Queuer):
    # TODO: First draft, needs to be adapted
    QUEUE_FILE = """
#!/usr/bin/env bash
##############################
#$ -cwd
#$ -V
#$ -q {queue}
#$ -N {jobname}
#$ -t 1-{array_range}
#$ -tc {concurrent}
#$ -l cuda=1
#$ -l "gputype=P100G*"
#$ -S /bin/bash
#$ -e {jobname}.log
#$ -o {jobname}.log
#$ -r n
#$ -sync y
##############################

if [ -d "/temp" ]; then
    TMPDIR=/temp/gastegger/
else
    TMPDIR=/tmp/gastegger/
fi

echo "tmpdir: $TMPDIR"
echo "node: $HOSTNAME"

if [ ! -d "$TMPDIR" ]; then
    mkdir $TMPDIR
fi

cp {dataset} $TMPDIR/{dataset_basename} 

task_name={basename}_$(printf "%06d" $SGE_TASK_ID)

{field_schnet_path} with {compdir}/${{task_name}}.yaml dataset=$TMPDIR/{dataset_basename}

"""

    def __init__(self, queue, executable, dataset_path, concurrent=100, basename='model', cleanup=True):
        # TODO: Check whether issues arise due to race conflicts during copying
        super(SchNarcQueuer, self).__init__(queue, executable, concurrent=concurrent, basename=basename,
                                            cleanup=cleanup)
        self.dataset_path = os.path.abspath(dataset_path)

    def _create_submission_command(self, n_inputs, compdir, jobname):
        dataset_basename = os.path.basename(self.dataset_path)
        submission_command = self.QUEUE_FILE.format(queue=self.queue,
                                                    basename=self.basename,
                                                    dataset_basename=dataset_basename,
                                                    array_range=n_inputs,
                                                    concurrent=self.concurrent,
                                                    field_schnet_path=self.executable,
                                                    jobname=jobname,
                                                    compdir=compdir,
                                                    dataset=self.dataset_path)
        return submission_command
