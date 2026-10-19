import dataset.data_loader.BaseLoader
try:
    import dataset.data_loader.COHFACELoader
except ImportError:
    pass
try:
    import dataset.data_loader.UBFCrPPGLoader
except ImportError:
    pass
try:
    import dataset.data_loader.PURELoader
except ImportError:
    pass
try:
    import dataset.data_loader.iBVPLoader
except ImportError:
    pass
try:
    import dataset.data_loader.SCAMPSLoader
except ImportError:
    pass
import dataset.data_loader.MMPDLoader
try:
    import dataset.data_loader.BP4DPlusLoader
except ImportError:
    pass
try:
    import dataset.data_loader.BP4DPlusBigSmallLoader
except ImportError:
    pass
try:
    import dataset.data_loader.UBFCPHYSLoader
except ImportError:
    pass
try:
    import dataset.data_loader.LADHLoader
except ImportError:
    pass
try:
    import dataset.data_loader.SUMSLoader
except ImportError:
    pass
try:
    import dataset.data_loader.MCDrPPGLoader
except ImportError:
    pass
