class PsmWriter:
    @staticmethod
    def write(frame, psms_all, file):
        """Writes a dictionary of PSMs to a file"""
        psm_strs = [
            f"{frame}, {ID}, {psm.value}, {psm.confidence}"
            for ID, psms in psms_all.items()
            for psm in psms
        ]
        with open(file, "a") as f:
            f.write("\n".join(psm_strs))
            if len(psm_strs) > 0:
                f.write("\n")
