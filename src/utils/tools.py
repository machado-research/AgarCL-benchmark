import sys
import PyExpUtils.utils.path as Path

def parseCmdLineArgs():
    path = sys.argv[0]
    path = Path.up(path)
    save = False
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        save = True

    save_type = 'png'
    if len(sys.argv) > 2:
        save_type = sys.argv[2]

    return (path, save, save_type)