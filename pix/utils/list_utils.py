def list_cat(ll):
    '''
    joint a iterable of lists into a single list.
    '''
    return sum(ll, [])

def list_remove(ll, fun):
    ''' 
    Remove elements of list, if fun(ele)==True.
    '''
    return [ele for ele in ll if not fun(ele)]