'''checkpoint manager '''
import os
import stat

class CheckpointManager:
    """Manage checkpoint files according to train_config of checkpoint."""

    def __init__(self):
        self._ckpoint_filelist = []

    @property
    def ckpoint_filelist(self):
        """Get all the related checkpoint files managed here."""
        return self._ckpoint_filelist

    @property
    def ckpoint_num(self):
        """Get the number of the related checkpoint files managed here."""
        return len(self._ckpoint_filelist)

    def update_ckpoint_filelist(self, directory, prefix):
        """Update the checkpoint file list."""
        self._ckpoint_filelist = []
        files = os.listdir(directory)
        for filename in files:
            if os.path.splitext(filename)[-1] == ".ckpt" and filename.startswith(prefix + "-"):
                mid_name = filename[len(prefix):-5]
                flag = not (True in [char.isalpha() for char in mid_name])
                if flag:
                    self._ckpoint_filelist.append(os.path.join(directory, filename))

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
            self._ckpoint_filelist.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def remove_oldest_ckpoint_file(self):
        """Remove the oldest checkpoint file from this checkpoint manager and also from the directory."""
        ckpoint_files = sorted(self._ckpoint_filelist, key=os.path.getmtime)
        self.remove_ckpoint_file(ckpoint_files[0])

    def keep_one_ckpoint_per_minutes(self, minutes, cur_time):
        """Only keep the latest one ckpt file per minutes, remove other files generated in [last_time, cur_time]."""
        del_list = []
        oldest_file = ''
        oldest_time = cur_time
        for ck_file in self._ckpoint_filelist:
            modify_time = os.path.getmtime(ck_file)
            if cur_time - modify_time < 60 * minutes:
                del_list.append(ck_file)

                if modify_time < oldest_time:
                    oldest_time = modify_time
                    oldest_file = ck_file

        for mv_file in del_list:
            if mv_file == oldest_file:
                continue
            self.remove_ckpoint_file(mv_file)
