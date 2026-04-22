def cart2orca(species,
              C,
              charge=0,
              multiplicity=1,
              filename=None,
              setting=None,
              mode='sp',
              new_gto=None):
    """convert cartesian_coords into orca input.
  mode:'sp','engrad','opt','freq'
  """
    pure_XYZ = cart2XYZ(species, C, header=False)
    XYZ = ''
    if new_gto is not None:
        xyz_lines = pure_XYZ.split('\n')
        for i in range(len(species)):
            XYZ += "{0} {1}\n".format(xyz_lines[i], new_gto[i])
    else:
        XYZ = pure_XYZ
    orca_inp = ''
    if setting is None:
        mode = mode.lower()
        if mode == 'sp':
            setting = '! B97-3c MINIPRINT\n'
        elif mode == 'engrad':
            setting = '! ENGRAD B97-3c MINIPRINT\n'
        elif mode == 'opt':
            setting = '! Opt B97-3c TightSCF MINIPRINT\n'
        elif mode == 'freq':
            setting = '! Opt Freq B97-3c TightSCF Grid5 NoFinalGrid MINIPRINT\n'
        else:
            raise KeyError(
                "choose a mode in {sp,opt,freq}, or give a input setting")
    for element in species:
        if ":" in element:
            setting = f'{setting[:-1]} Pmodel\n'
            break
    orca_inp += setting
    # print(setting)
    orca_inp += "* xyz {0} {1}\n".format(charge, multiplicity)
    orca_inp += XYZ
    orca_inp += "*\n"
    if filename:
        with open(filename, 'w') as f:
            f.write(orca_inp)
    else:
        return orca_inp
