import subprocess
import paramiko
import time
from typing import Union, List, Optional


class Popen:
    def __init__(
        self,
        args: Union[str, List[str]],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        hostname: Optional[str] = None,
        port: int = 22,
        username: Optional[str] = None,
        key_filename: Optional[str] = None,
    ):
        self.args = args
        self.cwd = cwd
        self.env = env
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.returncode: Optional[int] = None
        self._channel: Optional[paramiko.Channel] = None

        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(
            hostname=hostname,
            username=username,
            port=port,
            key_filename=key_filename,
            timeout=10,
        )

        # 构造命令
        command = self._build_command(args, cwd, env)

        # 执行命令
        transport = self._ssh.get_transport()
        transport.set_keepalive(1)
        self._channel = transport.open_session()
        self._channel.get_pty()
        if env:
            self._channel.update_environment(self.env)
        self._channel.exec_command(command)

        # 设置输入输出
        self.stdin = self._channel.makefile_stdin("wb")
        self.stdout = self._channel.makefile("r")
        self.stderr = self._channel.makefile_stderr("r")

    def _build_command(
        self, args: Union[str, List[str]], cwd: Optional[str], env: dict | None
    ) -> str:
        # 处理命令参数
        if isinstance(args, list):
            command = " ".join(args)
        else:
            command = args

        # 处理工作目录
        if cwd:
            command = f"cd {cwd} && {command}"

        # 处理环境变量
        env_str = ""
        if env:
            env_str = (
                " && ".join([f"export {k}='{v}'" for k, v in env.items()]) + " && "
            )
            command = f"{env_str}{command}"
        print(command)
        return command

    def poll(self) -> Optional[int]:
        if self.returncode is None and self._channel.exit_status_ready():
            self.returncode = self._channel.recv_exit_status()
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        start = time.time()
        while self.poll() is None:
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError("Command timed out")
            time.sleep(0.1)
        return self.returncode

    def terminate(self) -> None:
        if self._channel:
            self._channel.close()

    def kill(self) -> None:
        self.terminate()

    def __del__(self):
        self._ssh.close()

    # 兼容subprocess.Popen的属性
    # SSH协议不直接暴露PID
    @property
    def pid(self) -> Optional[int]:
        return None


if __name__ == "__main__":
    remote_process = Popen(
        "ls -la && echo $HELLO",
        "/nvme",
        {"HELLO": "WORLD"},
        "10.254.0.9",
        22222,
        "root",
        "/root/.ssh/id_rsa",
    )
    for line in remote_process.stdout:
        print(line.strip())
    remote_process.wait()

    local_process = subprocess.Popen(
        ["ls", "-la"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in local_process.stdout:
        print(line.strip())
    local_process.wait()
