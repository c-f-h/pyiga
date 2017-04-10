def test_bspline_cy():
    from pyiga import bspline, geometry, assemble_tools_cy
    kv = bspline.make_knots(3, 0.0, 1.0, 10)
    geo = geometry.bspline_quarter_annulus()
    asm = assemble_tools_cy.StiffnessAssembler2D((kv,kv), geo=geo)
