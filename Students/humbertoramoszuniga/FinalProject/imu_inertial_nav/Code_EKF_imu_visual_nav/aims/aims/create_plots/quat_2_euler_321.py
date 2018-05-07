
def quat_2_euler_321(quat):
    """
    Convert a quaternion to euler axis rotation sequence. 
    The sequence is angles a1-a2-a3 about the body-fixed z-y-x axes. 
    """
    
    # get DCM
    C = quat.asDCM()
    
    # get useful elements of C
    C02 = C[0,2]
    C12 = C[1,2]
    C00 = C[0,0]
    
    # calculate each angle
    a2 = arcsin(-C02)
    a1 = arccos(C00/cos(a2))
    a3 = arcsin(C12/cos(a2))
    
    # stack angles into array
    euler_321 = np.array([a1,a2,a3])
        
    return euler_321