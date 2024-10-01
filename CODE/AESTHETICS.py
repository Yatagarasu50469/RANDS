#==================================================================
#AESTHETICS
#==================================================================

#Quick print for titles in UI 
def sectionTitle(title):
    print()
    print((' ' * int((int(consoleColumns)-len(title))//2))+title)
    print(('¯' * int(consoleColumns)))

#Construct and print a header information for the running configuration
def programTitle(versionNum, configFileName):
    
    #Specify/generate program header information
    programName = "Risk Assessment Network for Dynamic Sampling"
    programName_length = len(programName)
    licenseHeader= "Licensed under GNU General Public License v3"
    licenseHeader_length = len(licenseHeader)
    versionHeader = "Program Version: " + versionNum
    versionHeader_length = len(versionHeader)
    
    #Offset needed to center header information
    maxLength = int(np.max([programName_length, licenseHeader_length, versionHeader_length]))
    programName_offset = (' ' * int((maxLength-programName_length)//2))
    licenseHeader_offset = (' ' * int((maxLength-licenseHeader_length)//2))
    versionHeader_offset = (' ' * int((maxLength-versionHeader_length)//2))
    
    #Generate and output the program header
    header = "\
"+"     _______   ______  __    __ _______   ______\n\
"+"|       \ /      \|  \  |  \       \ /      \ \n\
"+"| ▓▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓\ | ▓▓ ▓▓▓▓▓▓▓\  ▓▓▓▓▓▓\n\
"+"| ▓▓__| ▓▓ ▓▓__| ▓▓ ▓▓▓\| ▓▓ ▓▓  | ▓▓ ▓▓___\▓▓\n\
"+"| ▓▓    ▓▓ ▓▓    ▓▓ ▓▓▓▓\ ▓▓ ▓▓  | ▓▓\▓▓    \    " + programName_offset + programName + "\n\
"+"| ▓▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓ ▓▓\▓▓ ▓▓ ▓▓  | ▓▓_\▓▓▓▓▓▓    " + licenseHeader_offset + licenseHeader + "\n\
"+"| ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓ \▓▓▓▓ ▓▓__/ ▓▓  \__| ▓▓   " + versionHeader_offset + versionHeader + "\n\
"+"| ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓  \▓▓▓ ▓▓    ▓▓\▓▓    ▓▓\n\
"+" \▓▓   \▓▓\▓▓   \▓▓\▓▓   \▓▓\▓▓▓▓▓▓▓  \▓▓▓▓▓▓ "
    print(header)
    configHeader = "CONFIGURATION: " + os.path.splitext(os.path.basename(configFileName))[0].split('CONFIG_')[1]
    print()
    print((' ' * int((int(consoleColumns)-len(configHeader))//2))+configHeader)
    print(('=' * int(consoleColumns)))

#Determine console size
if systemOS != 'Windows':
    consoleRows, consoleColumns = os.popen('stty size', 'r').read().split()
elif systemOS == 'Windows':
    h = windll.kernel32.GetStdHandle(-12)
    csbi = create_string_buffer(22)
    res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
    (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
    consoleRows = bottom-top
    consoleColumns = right-left
