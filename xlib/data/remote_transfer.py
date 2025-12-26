import multiprocessing as mp
import os

import paramiko


def remote_is_exist(sftp, remote_path):
    try:
        sftp.stat(remote_path)
        return True
    except FileNotFoundError:
        return False
    
class RemoteTransfer:
    def __init__(self, hostname, port, username, password=None, key_filepath=None, key_type='rsa'):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.key_filepath = key_filepath
        self.key_type = key_type
        
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if key_filepath:
            if key_type == 'rsa':
                pkey = paramiko.RSAKey.from_private_key_file(key_filepath)
            elif key_type == 'ecdsa':
                pkey = paramiko.ECDSAKey.from_private_key_file(key_filepath)
            elif key_type == 'ed25519':
                pkey = paramiko.Ed25519Key.from_private_key_file(key_filepath)
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
            self.ssh.connect(hostname, port, username=username, pkey=pkey, look_for_keys=False, allow_agent=False)
        else:
            self.ssh.connect(hostname, port, username=username, password=password)
        print("[RemoteTransfer]: Connected")
        self.sftp = self.ssh.open_sftp()
        self.process = None
    
    def start_upload_process(self):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._upload_worker, args=(self.queue,))
        self.process.start()
        
    def _upload_worker(self, queue):
        upload_ssh = paramiko.SSHClient()
        upload_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if self.key_filepath:
            if self.key_type == 'rsa':
                pkey = paramiko.RSAKey.from_private_key_file(self.key_filepath)
            elif self.key_type == 'ecdsa':
                pkey = paramiko.ECDSAKey.from_private_key_file(self.key_filepath)
            elif self.key_type == 'ed25519':
                pkey = paramiko.Ed25519Key.from_private_key_file(self.key_filepath)
            else:
                raise ValueError(f"Unsupported key type: {self.key_type}")
            upload_ssh.connect(self.hostname, self.port, username=self.username, pkey=pkey, look_for_keys=False, allow_agent=False)
        else:
            upload_ssh.connect(self.hostname, self.port, username=self.username, password=self.password)
        upload_sftp = upload_ssh.open_sftp()
        while self.process.is_alive():
            task = queue.get()
            if task is None:
                break
            local_file, remote_file, overwrite = task
            remote_tmp = os.path.join(os.path.dirname(remote_file), f".{os.path.basename(remote_file)}.tmp")
            if remote_is_exist(upload_sftp, remote_file):
                if overwrite:
                    upload_sftp.remove(remote_file)
                else:
                    print(f"[RemoteTransfer]: File {remote_file} already exists. Skipping upload.")
                    continue
            if remote_is_exist(upload_sftp, remote_file):
                upload_sftp.remove(remote_tmp)
            upload_sftp.put(local_file, remote_tmp)
            upload_sftp.rename(remote_tmp, remote_file)
            
    def upload_folder(self, local_folder, remote_folder, overwrite=False):
        try:
            self.sftp.listdir(remote_folder)
        except IOError:
            self._mkdir_p(remote_folder)

        for item in os.listdir(local_folder):
            local_path = os.path.join(local_folder, item)
            remote_path = os.path.join(remote_folder, item)

            if os.path.isdir(local_path):
                self.upload_folder(local_path, remote_path, overwrite)
            else:
                self.upload_file(local_path, remote_path, overwrite)

    def _mkdir_p(self, remote_directory):
        dirs = []
        dir_path = remote_directory
        while len(dir_path) > 1:
            dirs.append(dir_path)
            dir_path, _ = os.path.split(dir_path)
        dirs.reverse()
        for d in dirs:
            try:
                self.sftp.listdir(d)
            except IOError:
                self.sftp.mkdir(d)
                
    def list_remote_dir(self, remote_dir):
        return self.sftp.listdir(remote_dir)

    def upload_file(self, local_file, remote_file, overwrite=False):
        if self.process is None:
            remote_tmp = os.path.join(os.path.dirname(remote_file), f".{os.path.basename(remote_file)}.tmp")
            if remote_is_exist(self.sftp, remote_file):
                if overwrite:
                    self.sftp.remove(remote_file)
                else:
                    print(f"[RemoteTransfer]: File {remote_file} already exists. Skipping upload.")
                    return
            if remote_is_exist(self.sftp, remote_file):
                self.sftp.remove(remote_tmp)
            self.sftp.put(local_file, remote_tmp)
            self.sftp.rename(remote_tmp, remote_file)
        else:
            self.queue.put((local_file, remote_file, overwrite))

    def download_file(self, remote_file, local_file, overwrite=False):
        temp_local_path = os.path.join(os.path.dirname(local_file), f".{os.path.basename(local_file)}.tmp")
        if os.path.exists(local_file):
            if overwrite:
                os.remove(local_file)
            else:
                print(f"[RemoteTransfer]: File {local_file} already exists. Skipping download.")
                return
        if os.path.exists(temp_local_path):
            os.remove(temp_local_path)

        self.sftp.get(remote_file, temp_local_path)
        os.rename(temp_local_path, local_file)
        
    def download_folder(self, remote_folder, local_folder, overwrite=False):
        for item in self.sftp.listdir(remote_folder):
            remote_path = os.path.join(remote_folder, item)
            local_download_folder = os.path.join(local_folder, os.path.basename(remote_folder))
            if not os.path.exists(local_download_folder):
                os.makedirs(local_download_folder, exist_ok=True)
            
            if self.sftp.stat(remote_path).st_mode & 0o40000:
                self.download_folder(remote_path, local_download_folder, overwrite=overwrite)
            else:
                local_path = os.path.join(local_download_folder, item)
                self.download_file(remote_path, local_path, overwrite=overwrite)

    def close(self):
        self.sftp.close()
        self.ssh.close()
        if self.process is not None:
            self.queue.put(None)
            self.process.join()
