import os

from avstack.config import HOOKS
from avstack.utils.logging import Logger


@HOOKS.register_module()
class TrustLogger(Logger):
    prefix = "trust"
    file_ending = "txt"

    def _get_file_name(self, trust_out, *args, **kwargs):
        file = os.path.join(
            self.output_folder,
            f"{self.prefix}-{trust_out.frame:010d}-{trust_out.timestamp:012.2f}.{self.file_ending}",
        )
        return file

    def _encode(self, trust_out, *args, **kwargs):
        return trust_out.encode()
