import io
import os
import tarfile
from typing import Tuple

import docker
import docker.models.containers


class Container:
    def __init__(self, image: str, name: str | None = None, user: str = 'root') -> None:
        self.client = docker.from_env()
        self.image_name = image
        self.image = self.client.images.get(image)
        self.container: docker.models.containers.Container = self.client.containers.run(
            self.image,
            name=name,
            detach=True,
            tty=True,
        )
        self.user = user

    def __del__(self):
        if hasattr(self, 'container'):
            try:
                self.container.kill()
                self.container.remove(force=True)
            except Exception:
                # Ignore errors during cleanup
                pass

    def exec_cmd(self, cmd: str, workdir: str | None = None) -> Tuple[int, str, str]:
        exit_code, (stdout, stderr) = self.container.exec_run(
            cmd,
            user=self.user,
            workdir=workdir,
            stdout=True,
            stderr=True,
            demux=True,
        )
        stdout = stdout.decode() if stdout else ''
        stderr = stderr.decode() if stderr else ''
        return exit_code, stdout, stderr

    def copy_from(self, src: str, dst: str) -> None:
        bits, stat = self.container.get_archive(src)
        if stat["size"] == 0:
            raise FileNotFoundError(f"File {src} not found in the container")

        tarstream = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=tarstream, mode="r") as tar:
            tar.extractall(os.path.dirname(dst))

    def copy_to(self, src: str, dst: str) -> None:
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tar.add(src, arcname=os.path.basename(src))
        tarstream.seek(0)
        self.container.put_archive(os.path.dirname(dst), tarstream)
