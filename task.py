import sys


class Task(object):
    def __init__(self, formula=None):
        # l_0_0_0: regions, type, #robots, indicator for same robots
        # self.formula = '<> l1_1_1_0 && <> l3_1_1_0' #  && <> l4_4_2_0 && <> l5_2_4_0 && <> l6_3_3_0 && <> l7_4_4_0'
                       # '&& <> l8_4_3_0 && <> l9_5_3_0 && <> l10_1_2_0'
        # self.ap = [[['l1', 1, 1]], [['l2', 1, 2]]]
        # , [['l3', 3, 3]], [['l4', 4, 2]]]
        # n = int(sys.argv[2])//3
        # n = 2
        # self.formula = '[] <> l1_1_{0}_0 && []<> l2_2_{0}_0 && [] <> l3_3_{0}_0  && []<> l4_4_{0}_0 && [] <> l5_5_{0}_0 && ' \
        #                '<> [] l6_1_{0}_0 && <> [] l7_2_{0}_0 && <>[] l8_3_{0}_0 && <> [] l9_4_{0}_0 && <>[] l10_5_{0}_0 '.format(n)
        # self.formula = '[]<> l1_1_{0}_0 &&  []<> l2_2_{0}_0 && <> l3_3_{0}_0' \
        #                ' && <> l6_1_{0}_0 && <>  l7_2_{0}_0 &&  <> l8_3_{0}_0'.format(n)

        # self.formula = '[] <> l1_1_{0}_0 && <> [] l6_1_{0}_0'.format(n)
        # self.formula = '[] <> l3_2_2 && [] <> l4_1_2 && [] <> l5_3_2  && [] <> l6_1_3 && []<> l7_1_3 && ' \
        #                '[]<> l1_2_2 && []<> l2_3_2'
        # sel f.formula = '[] <> l6_1_2 && []<> l4_1_2 && []<> l2_1_2'
        # && <> l5_2_1 && <> l6_3_1 && <> l7_1_1 ' \
        #                '&& <> l8_2_1 && <> l9_3_1 && <> l10_1_1'
        # self.formula = '<> l1_1_1 && <> l2_2_1 && <> ((l3_1_1 && l2_1_1) || l3_2_1)  && <> l4_1_1 && <> l5_1_1'
        # self.formula = '<> (l1_1_1_0 && <> l2_1_1_0) && <> ((l3_1_1_0 && l5_1_2_0) && l4_1_1_0))'
        # self.formula = '<> (l1_1_3 && X <> l2_2_4) && <> (l3_1_3  && l4_3_2)'
        # self.formula = '<> (l1_1_1 && X <> (l2_1_2 && X <> l3_2_1))'
        # self.formula = '<> (l1_1_1_0 && l5_3_1_0) && <> l2_2_2_0 && <> l4_3_2_0'
        # self.ap = [[['l1', 1, 1], ['l5', 3, 1]], [['l2', 2, 2]], [['l4', 3, 2]]]
        # self.formula = '<> (l1_1_2_0 && X<> l2_1_1_0) && <> l3_1_1_0'
        # self.formula = '[]<> (l1_1_1 && <> l2_1_1) && <> [] l3_1_1'
        # self.formula = '[]<> (l1_1_1 && <> l2_1_1) && [] <> ((l3_1_1 && l5_1_2) || l4_1_1))'
        # self.formula = '[] <> (l1_1_1 && <> ((l2_1_1 && l2_1_1) || l3_2_1)) && <> [] l2_2_1'
        # self.formula = '[] <> l1_1_1 && [] <> l2_2_1 '
        # self.formula = '[] <> (l1_1_1_0 && <> l2_1_1_0)'
        # self.formula = '<> (l1_1_2_0 && <> l3_1_1_0) && <> l2_2_1_0'
        # self.formula = '[]<> l1_1_2 && <> (l2_2_2 || l3_2_2) && []<> l3_1_2'
        # self.formula = '[] <> l1_1_1_0 && <> l3_1_1_0  && [] <> (l2_2_1_0 || l3_2_1_0) '  # ok
        # self.formula = '[]<> l1_1_1_1 && []<> l3_1_1_1 && []<> (l2_2_1_0 || l3_2_1_0) '  # not ok
        # self.formula = '[] <> l1_1_1_1 && <> l3_1_1_1  && [] <> (l3_2_1_0) '  # ok
        # self.formula = '[] <> l1_1_1_1 && [] <> (l2_2_1_0 && l3_2_2_0) '
        # self.formula = '[] <> (l2_2_1_0 || l3_2_1_0)'  # ok
        # self.formula = '<> (l1_3_1_0 && l1_1_1_0) && <> l4_3_2_0 && <> l3_1_2_0 && ! l4_3_2_0 U l3_1_2_0 && [] ! l5_1_1_0'
        # self.formula = '<> (l10_1_2_0 && <> l3_1_1_0) && ! l3_1_1_0 U l10_1_2_0 && [] ! l2_1_1_0 '  # && [] ! l8_1_1_0 && []! l9_1_1_0
        # self.ap = [[['l10', 1, 2]], [['l3', 1, 1]]]
        # self.formula = '<> l2_1_4_0 && <> l3_1_2_0 && <> l4_1_2_0'  # && [] ! l8_1_1_0 && []! l9_1_1_0
        # self.ap = [[['l2', 1, 4]], [['l3', 1, 2]], [['l4', 1, 2]]]
        # self.formula = '<> l2_1_8_0 && <> l3_1_4_0 && <> l4_1_4_0'  # && [] ! l8_1_1_0 && []! l9_1_1_0
        # self.ap = [[['l2', 1, 8]], [['l3', 1, 4]], [['l4', 1, 4]]]
        # self.formula = '<> l2_1_4_0 && <> l3_1_2_0 && <> l4_1_2_0 && !l4_1_1_0 U (l5_1_1_0 && l6_1_1_0) '
        # self.ap = [[['l2', 1, 12]], [['l3', 1, 6]], [['l4', 1, 6]]]
        # self.formula = '<> (l2_2_3_1 && <> l3_2_3_1) && <> l3_1_3_0 &&  <> l4_1_1_0 '
        # self.formula = '<> l6_1_2_1 && <> l5_1_2_1 && <>l3_1_3_0 && !l5_1_1_0 U l6_1_2_1 && !l3_1_1_0 U l5_1_2_1'
        # self.formula = '<> (l4_2_1_1 && (l4_2_1_1 U l3_2_1_0) && <> l2_2_1_1)'
        # self.formula = '<> l2_2_3_0'
        # self.formula = '<> l4_1_1_0'

        # self.formula = '<> (l2_1_2_1 && X <> l3_1_2_1) && <> l4_2_1_0 && ! l3_1_2_0 U l4_2_1_0'
        # && [] ! l8_1_1_0 && []! l9_1_1_0
        # -----------------test strong implication -------------
        # self.formula = '<> (l2_1_1_0 && !l3_1_1) && [] <> l2_1_1_0'
        # self.formula = '[] <> (l2_1_1_0 || l3_1_1_0)'
        # self.formula = '<> (l2_1_1_0 || l4_1_1_0) && []! l3_1_1'
        # self.formula = '<> (l2_1_1_0 || l4_1_1_0) && []! l3_1_1 && [] <> (l2_1_1_0 || l4_1_1_0)'
        # self.formula = '[] <> (!l3_1_1 || !l4_1_1)'
        # self.formula = '[] <> (!l3_1_1 || l4_1_1_0 || !l5_1_1 )'
        # self.formula = '<> !l3_1_1'


        # ----------------- test prunging steps ----------------
        # self.formula = '<> (l1_1_2_0 && !l1_1_1 && !l1_1_2) && <> l2_2_1_0'
        # self.formula = 'l1_1_2_0'
        # self.formula = '(l1_1_2_0 && !l1_1_5_0) U l2_1_1_0'
        # ----------------- test the poset mining ---------------
        # self.formula = '[] <> (l2_1_1_0 && <> l4_1_1_0)'
        # self.formula = '<> l2_1_1_0 && <> l4_1_1_0 && l1_1_1_0 U l4_1_1_0'
        # self.formula = '<> (l4_1_2_1 && l3_1_1_0 && !l4_2_1) && <> l5_1_2_1 && [] !l2_1_1'
        # self.formula = '<> (l3_1_2_1 && <> l4_1_2_1)'
        # self.formula = '[] <> (l3_1_2_1 && <> l4_1_2_1) && <> l5_2_1_0'
        # self.formula = '[] <> (l4_1_2_1 &&  <> l3_1_2_1) && <> l4_2_1_0 && ! l3_1_2 U l4_2_1_0'
        # self.formula = '(l1_1_2_0 && !l2_1_1) U (l2_1_2_0 && ! l1_2_1 && !l1_1_1)'
        # self.formula = '<> l2_1_1_0 && <> l4_2_1_0 && []!(l2_1_1 && l4_2_1)'
        # self.formula = '<> (l2_1_1_0 ||  ! l3_1_1)'
        # self.formula = '(l1_1_2_0) U (l2_1_2_0 )'
        # self.formula = '[]<> l2_1_12_0 && []<> l3_1_6_0 && [] <> l4_1_6_0 && (!l4_1_1 U (l5_1_1_0 && l6_1_1_0))'
        # self.formula = '[] <> (l4_1_1_1 && <> l3_1_1_1) && []! l2_1_1'
        # self.formula = '<> l2_1_1_1 && [] <> (l4_1_1_1  && l3_1_1_0)'
        # self.formula = '[] <> (l4_1_2_1 && <> l3_1_2_1) && []! l5_1_2 && <> [] l6_2_2_0'
        self.formula = '[] <> (l4_1_4_1 && <> l3_1_4_1) && <> [] l2_2_4_0'
        # self.formula = '<>[] l6_1_1_0 && [] <> l5_1_2_0 && []<> l3_1_3_1 && []<>(l4_1_3_1 && (l4_1_3_1 U l2_2_4_0))' \
        #                ' && []<> l3_2_4_0'
        # self.formula = '[] <> ((l2_1_2_0 && ! l3_1_1) || (l3_1_2_0 && ! l2_1_1))'
        # self.formula = '[]<> l3_1_3_1 && []<>(l4_1_3_1 && (l4_1_3_1 U l2_2_4_0))'
        # self.formula = '[] <> l3_1_2_0 &&  [] <> l3_2_2_0'
        # self.formula = '[] <> (l2_1_2_0 && <> l4_1_4_0)'
        # self.formula = '[]<>(l3_1_1_0 && !l1_1_1)'
        # self.formula = '[] <> (l4_1_1_0 && <> l5_1_1_0) &&  [] ! (l4_1_1_0 && l5_1_1_0)'
        # self.formula = '<> (l4_1_1_1 && X (l4_1_1_1 U l5_1_1_0)) && [] <> l6_1_1_1'
        # self.formula = 'l1_1_3_1 && <> l4_1_3_1'
        # self.formula = '<> (!l3_1_1 || l4_1_2_0)'
        # self.formula = '<> ((l2_1_2_1 && ! l3_1_2) && <> l3_1_2_1) && <> l4_2_1_0 && (! l3_1_2 U l4_2_1_0)'
        # self.formula = '[]l1_1_4_0  && <> l2_2_1_0 && <> l4_2_1_0'
        # self.formula = '[]<> (l3_1_2_0 && ! l4_1_1) && [] <> (l4_1_2_0 && ! l3_1_1)'
        # self.formula = '(l1_1_4_0 && ! l4_1_1) U (l4_2_4_0 && !l1_1_1)'
        # self.formula = '<> l2_1_1_1 && <> l4_1_1_1'
        # self.formula = '<> (l1_1_1_1 && l1_1_1_1 U l4_2_1_0)'
        # self.formula = '<> (l2_1_1_0 && l2_1_1_0 U l3_1_1_0)'

        self.ap = [[['l3', 1, 2]], [['l2', 2, 1]]]
        if formula:
            self.formula = formula
        # case 3
        # self.formula = '[]<> (l1_1_3_1 && l4_4_4_0)  && []<> l2_3_3_2 && [] <> l3_1_3_1 && [] <> l5_3_3_2 && []<> l6_4_2_0' \
        #                '&& <> (l7_4_3_0 && (l8_5_3_3 || l10_5_3_3) && X <> l9_5_3_3) && <>[] l10_5_1_0'
        # self.formula = '<> (l7_4_3_0 && (l8_5_3_3 || l10_5_3_3) && X <> l9_5_3_3) && <>[] l10_5_1_0'
        # self.ap = [[['l1', 1, 3], ['l4', 4, 4]], [['l2', 3, 3]], [['l3', 1, 3]], [['l5', 3, 3]],
        #            [ ['l6', 4, 2]], [['l7', 4, 3], ['l8', 5, 3]], [['l7', 4, 3], ['l10', 5, 3]], [['l9', 5, 3]],
        #            [['l10', 5, 1]]]

