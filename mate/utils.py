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


class BetaDistWriter:
    @staticmethod
    def write(frame, trusts, file):
        """Writes a dictionary of Beta distributions to a file"""
        dist_strs = [
            f"{frame}, {ID}, {dist.mean:.4f}, {dist.variance:.4f}, "
            f"{dist.alpha:.4f}, {dist.beta:.4f}"
            for ID, dist in trusts.items()
        ]
        with open(file, "a") as f:
            f.write("\n".join(dist_strs))
            if len(dist_strs) > 0:
                f.write("\n")
